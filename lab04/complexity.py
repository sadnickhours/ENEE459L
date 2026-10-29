from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}


BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

BUFFER_DTYPE = "fp32"

MIN_MODELS_FOR_FIT = 3
TIE_EXACT = True


# ===========================================================================
# 1. Counting learnable parameters
# ===========================================================================


def count_parameters(graph: Graph) -> dict[str, Any]:
    """Count learnable parameters in every layer and in the graph."""

    per_layer = {}
    total = 0

    for ly in graph.layers:
        per_layer[ly.name] = _layer_parameters(ly)
        total += per_layer[ly.name]

    return computed(
        value=total,
        source="count_parameters",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


def _layer_parameters(ly: Layer) -> int:
    """Return the number of learnable parameters in one layer."""

    if ly.kind == "conv":
        cout = ly.out_shape[0]
        cin = ly.in_shape[0]

        if ly.kernel is None:
            kh, kw = 1, 1
        else:
            kh, kw = ly.kernel

        nweights = cout * (cin // ly.groups) * kh * kw

        if ly.bias:
            nweights += cout

        return nweights

    elif ly.kind == "linear":
        fout = ly.out_shape[0]
        fin = ly.in_shape[0]

        nweights = fout * fin

        if ly.bias:
            nweights += fout

        return nweights

    elif ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]

    else:
        return 0


# ===========================================================================
# 2. Stored model size
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Calculate stored parameter and persistent-buffer memory in bytes."""

    per_dtype = {}
    per_layer = {}
    buffer_bytes = 0.0

    for ly in graph.layers:
        # Learnable parameters
        n = _layer_parameters(ly)

        param_bytes = n * dtype_bytes(ly.weight_dtype)

        per_dtype[ly.weight_dtype] = (
            per_dtype.get(ly.weight_dtype, 0.0) + param_bytes
        )

        layer_bytes = param_bytes

        # Batch-normalization running mean and variance
        if ly.kind == "bn":
            buffers = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]

            buff_bytes = buffers * dtype_bytes(BUFFER_DTYPE)

            buffer_bytes += buff_bytes

            per_dtype[BUFFER_DTYPE] = (
                per_dtype.get(BUFFER_DTYPE, 0.0) + buff_bytes
            )

            layer_bytes += buff_bytes

        per_layer[ly.name] = layer_bytes

    total = sum(per_layer.values())

    return computed(
        value=total,
        source="model_size_bytes",
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
    )


# ===========================================================================
# 3. Activation footprint and liveness
# ===========================================================================


def count_activations(graph: Graph) -> dict[str, Any]:
    """Calculate total activation production and peak resident memory."""

    last_use = _last_use(graph)

    # Live tensors are tracked in bytes.
    live = {}

    input_bytes = (
        _elements(graph.input_shape)
        * dtype_bytes(graph.precision)
    )

    live["__input__"] = input_bytes

    total_elements = 0
    total_bytes = 0.0

    peak_bytes = sum(live.values())
    peak_at = "__input__"

    for i, ly in enumerate(graph.layers):
        out_elements = ly.out_elements
        out_b = out_elements * dtype_bytes(ly.act_dtype)

        # Add this layer's output to the live set.
        live[ly.name] = out_b

        # Total activation production excludes the network input.
        total_elements += out_elements
        total_bytes += out_b

        # Calculate current resident memory.
        resident = sum(live.values())

        if resident > peak_bytes:
            peak_bytes = resident
            peak_at = ly.name

        # Remove tensors whose final use occurs at this layer.
        to_remove = []

        for tensor, last_index in last_use.items():
            if last_index == i and tensor in live:
                to_remove.append(tensor)

        for tensor in to_remove:
            del live[tensor]

    peak_elements = _peak_elements(graph, last_use)

    return computed(
        value=peak_bytes,
        source="count_activations",
        peak_bytes=peak_bytes,
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
    )


def _elements(shape: tuple[int, ...]) -> int:
    """Return the number of elements represented by a shape."""

    elements = 1

    for dim in shape:
        elements *= dim

    return elements


def _last_use(graph: Graph) -> dict[str, int]:
    """Determine the last layer index at which each tensor is consumed."""

    last = {}

    for i, ly in enumerate(graph.layers):

        # In graph.py, an unspecified reads field is represented by ().
        # Therefore, use truthiness rather than `is not None`.
        if ly.reads:
            for tensor in ly.reads:
                last[tensor] = i

        else:
            if i == 0:
                # First layer consumes the network input.
                last["__input__"] = 0
            else:
                # Sequential layer consumes the previous layer's output.
                previous_name = graph.layers[i - 1].name
                last[previous_name] = i

        # If nobody consumes this layer's output, it dies where produced.
        last.setdefault(ly.name, i)

    # The final output remains alive until the end of the forward pass.
    if graph.layers:
        last[graph.layers[-1].name] = len(graph) - 1

    return last


def _peak_elements(
    graph: Graph,
    last_use: dict[str, int],
) -> int:
    """Calculate the maximum number of activation elements simultaneously live."""

    live = {
        "__input__": _elements(graph.input_shape)
    }

    peak = sum(live.values())

    for i, ly in enumerate(graph.layers):

        # Add this layer's output.
        live[ly.name] = ly.out_elements

        current = sum(live.values())

        if current > peak:
            peak = current

        # Remove tensors whose final use is this layer.
        to_remove = []

        for tensor, last_index in last_use.items():
            if last_index == i and tensor in live:
                to_remove.append(tensor)

        for tensor in to_remove:
            del live[tensor]

    return peak


# ===========================================================================
# 4. MAC to FLOP conversion
# ===========================================================================


def to_flops(
    macs: dict[str, Any],
    convention: str = "mac_is_two_flops",
) -> dict[str, Any]:
    """Convert a MAC finding into a FLOP finding."""

    # The MAC finding must contain a valid answer.
    if not is_answered(macs):
        return unknown(
            "to_flops",
            "no valid MAC count was provided",
        )

    # Do not silently assume a convention we do not recognize.
    if convention not in FLOP_CONVENTIONS:
        return unknown(
            "to_flops",
            f"unrecognized convention '{convention}'; "
            f"allowed options: {list(FLOP_CONVENTIONS.keys())}",
        )

    factor = FLOP_CONVENTIONS[convention]

    total_flops = macs["value"] * factor

    # Convert per-layer MAC counts if they are available.
    per_layer = {}

    if "per_layer" in macs:
        for layer_name, layer_macs in macs["per_layer"].items():
            per_layer[layer_name] = layer_macs * factor

    return computed(
        value=total_flops,
        source=macs["source"],
        convention=convention,
        flops_per_mac=factor,
        per_layer=per_layer,
    )
    
    
def _layer_macs(ly: Layer) -> int:
    """Estimate the number of multiply-accumulate operations for one layer."""

    if ly.kind == "conv":
        cin = ly.in_shape[0]
        cout = ly.out_shape[0]

        if ly.kernel is None:
            kh, kw = 1, 1
        else:
            kh, kw = ly.kernel

        cin_per_group = cin // ly.groups

        return ly.out_elements * cin_per_group * kh * kw

    elif ly.kind == "linear":
        return ly.in_elements * ly.out_elements

    else:
        return 0


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    """Calculate per-layer and whole-model arithmetic intensity."""

    per_layer = {}
    total_flops = 0
    total_bytes = 0.0

    for i, ly in enumerate(graph.layers):
        # Resolve the dtype of the incoming activation tensor.
        if ly.reads:
            producer = graph.get(ly.reads[0])
            in_dtype = (
                producer.act_dtype if producer is not None
                else graph.precision
            )
        elif i > 0:
            in_dtype = graph.layers[i - 1].act_dtype
        else:
            in_dtype = graph.precision

        # Input activation memory traffic.
        input_elements = ly.in_elements

        if ly.kind == "add" and len(ly.reads) == 2:
            input_elements *= 2

        input_bytes = input_elements * dtype_bytes(in_dtype)

        # Learnable parameters and persistent BatchNorm buffers.
        weight_bytes = (
            _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)
        )

        buffer_bytes = 0.0
        if ly.kind == "bn":
            buffer_bytes = (
                BN_BUFFERS_PER_CHANNEL
                * ly.out_shape[0]
                * dtype_bytes(BUFFER_DTYPE)
            )

        # Output activation memory traffic.
        output_bytes = (
            ly.out_elements * dtype_bytes(ly.act_dtype)
        )

        bytes_moved = (
            input_bytes + weight_bytes + buffer_bytes + output_bytes
        )

        # One MAC counts as two FLOPs.
        flops = _layer_macs(ly) * FLOPS_PER_MAC

        intensity = (
            round(flops / bytes_moved, 4)
            if bytes_moved != 0
            else 0.0
        )

        per_layer[ly.name] = {
            "flops": flops,
            "bytes": float(bytes_moved),
            "intensity": intensity,
        }

        total_flops += flops
        total_bytes += bytes_moved

    model_intensity = (
        round(total_flops / total_bytes, 4)
        if total_bytes != 0
        else 0.0
    )

    return computed(
        value=model_intensity,
        source="arithmetic_intensity",
        message=f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=float(total_bytes),
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )

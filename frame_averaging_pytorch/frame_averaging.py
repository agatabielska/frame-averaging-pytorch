from typing import Optional

from random import randrange

import torch
from torch.nn import Module


from einops import rearrange, repeat, reduce, einsum

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type, cast, Optional
from collections import namedtuple

"""
Contains utility functions for working with nested python data structures.

A *pytree* is Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_unflatten` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]

class NodeDef(NamedTuple):
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc

SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}

def _register_pytree_node(typ: Any, flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc) -> None:
    SUPPORTED_NODES[typ] = NodeDef(flatten_fn, unflatten_fn)

def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())

def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}

def _list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None

def _list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return list(values)

def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return list(d), None

def _tuple_unflatten(values: List[Any], context: Context) -> Tuple[Any, ...]:
    return tuple(values)

def _namedtuple_flatten(d: NamedTuple) -> Tuple[List[Any], Context]:
    return list(d), type(d)

def _namedtuple_unflatten(values: List[Any], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))

_register_pytree_node(dict, _dict_flatten, _dict_unflatten)
_register_pytree_node(list, _list_flatten, _list_unflatten)
_register_pytree_node(tuple, _tuple_flatten, _tuple_unflatten)
_register_pytree_node(namedtuple, _namedtuple_flatten, _namedtuple_unflatten)


# h/t https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
def _is_namedtuple_instance(pytree: Any) -> bool:
    typ = type(pytree)
    bases = typ.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(typ, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(entry) == str for entry in fields)

def _get_node_type(pytree: Any) -> Any:
    if _is_namedtuple_instance(pytree):
        return namedtuple
    return type(pytree)

# A leaf is defined as anything that is not a Node.
def _is_leaf(pytree: PyTree) -> bool:
    return _get_node_type(pytree) not in SUPPORTED_NODES.keys()


# A TreeSpec represents the structure of a pytree. It holds:
# "type": the type of root Node of the pytree
# context: some context that is useful in unflattening the pytree
# children_specs: specs for each child of the root Node
# num_leaves: the number of leaves
class TreeSpec:
    def __init__(self, typ: Any, context: Context, children_specs: List['TreeSpec']) -> None:
        self.type = typ
        self.context = context
        self.children_specs = children_specs
        self.num_leaves: int = sum([spec.num_leaves for spec in children_specs])

    def __repr__(self) -> str:
        return f'TreeSpec({self.type.__name__}, {self.context}, {self.children_specs})'

    def __eq__(self, other: Any) -> bool:
        result = self.type == other.type and self.context == other.context \
            and self.children_specs == other.children_specs \
            and self.num_leaves == other.num_leaves
        # This should really not be necessary, but mypy errors out without it.
        return cast(bool, result)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

class LeafSpec(TreeSpec):
    def __init__(self) -> None:
        super().__init__(None, None, [])
        self.num_leaves = 1

    def __repr__(self) -> str:
        return '*'

def tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree.
    """
    if _is_leaf(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    # Recursively flatten the children
    result : List[Any] = []
    children_specs : List['TreeSpec'] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def tree_unflatten(values: List[Any], spec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.
    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, TreeSpec):
        raise ValueError(
            f'tree_unflatten(values, spec): Expected `spec` to be instance of '
            f'TreeSpec but got item of type {type(spec)}.')
    if len(values) != spec.num_leaves:
        raise ValueError(
            f'tree_unflatten(values, spec): `values` has length {len(values)} '
            f'but the spec refers to a pytree that holds {spec.num_leaves} '
            f'items ({spec}).')
    if isinstance(spec, LeafSpec):
        return values[0]

    unflatten_fn = SUPPORTED_NODES[spec.type].unflatten_fn

    # Recursively unflatten the children
    start = 0
    end = 0
    child_pytrees = []
    for child_spec in spec.children_specs:
        end += child_spec.num_leaves
        child_pytrees.append(tree_unflatten(values[start:end], child_spec))
        start = end

    return unflatten_fn(child_pytrees, spec.context)

def tree_map(fn: Any, pytree: PyTree) -> PyTree:
    flat_args, spec = tree_flatten(pytree)
    return tree_unflatten([fn(i) for i in flat_args], spec)

# Broadcasts a pytree to the provided TreeSpec and returns the flattened
# values. If this is not possible, then this function returns None.
#
# For example, given pytree=0 and spec=TreeSpec(list, None, [LeafSpec(), LeafSpec()]),
# would return [0, 0]. This is useful for part of the vmap implementation:
# a user can pass in vmap(fn, in_dims)(*inputs). `in_dims` should be
# broadcastable to the tree structure of `inputs` and we use
# _broadcast_to_and_flatten to check this.
def _broadcast_to_and_flatten(pytree: PyTree, spec: TreeSpec) -> Optional[List[Any]]:
    assert isinstance(spec, TreeSpec)

    if _is_leaf(pytree):
        return [pytree] * spec.num_leaves
    if isinstance(spec, LeafSpec):
        return None
    node_type = _get_node_type(pytree)
    if node_type != spec.type:
        return None

    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, ctx = flatten_fn(pytree)

    # Check if the Node is different from the spec
    if len(child_pytrees) != len(spec.children_specs) or ctx != spec.context:
        return None

    # Recursively flatten the children
    result : List[Any] = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = _broadcast_to_and_flatten(child, child_spec)
        if flat is not None:
            result += flat
        else:
            return None

    return result

# main class

class FrameAverage(Module):
    def __init__(
        self,
        net: Optional[Module] = None,
        dim = 3,
        stochastic = False,
        invariant_output = False,
        return_stochastic_as_augmented_pos = False  # will simply return points as augmented points of same shape on forward
    ):
        super().__init__()
        self.net = net

        assert dim > 1

        self.dim = dim
        self.num_frames = 2 ** dim

        # frames are all permutations of the positive (+1) and negative (-1) eigenvectors for each dimension, iiuc
        # so there will be 2 ^ dim frames

        directions = torch.tensor([-1, 1])

        colon = slice(None)
        accum = []

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon

            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim = -1)
        operations = rearrange(operations, '... d -> (...) d')

        assert operations.shape == (self.num_frames, dim)

        self.register_buffer('operations', operations)

        # whether to use stochastic frame averaging
        # proposed in https://arxiv.org/abs/2305.05577
        # one frame is selected at random

        self.stochastic = stochastic
        self.return_stochastic_as_augmented_pos = return_stochastic_as_augmented_pos

        # invariant output setting

        self.invariant_output = invariant_output

    def forward(
        self,
        points,
        *args,
        frame_average_mask = None,
        return_framed_inputs_and_averaging_function = False,
        **kwargs,
    ):
        """
        b - batch
        n - sequence
        d - dimension (input or source)
        e - dimension (target)
        f - frames
        """

        assert points.shape[-1] == self.dim, f'expected points of dimension {self.dim}, but received {points.shape[-1]}'

        # account for variable lengthed points

        if exists(frame_average_mask):
            frame_average_mask = rearrange(frame_average_mask, '... -> ... 1')
            points = points * frame_average_mask

        # shape must end with (batch, seq, dim)

        batch, seq_dim, input_dim = points.shape

        # frame averaging logic

        if exists(frame_average_mask):
            num = reduce(points, 'b n d -> b 1 d', 'sum')
            den = reduce(frame_average_mask.float(), 'b n 1 -> b 1 1', 'sum')
            centroid = num / den.clamp(min = 1)
        else:
            centroid = reduce(points, 'b n d -> b 1 d', 'mean')

        centered_points = points - centroid

        if exists(frame_average_mask):
            centered_points = centered_points * frame_average_mask

        covariance = einsum(centered_points, centered_points, 'b n d, b n e -> b d e')

        _, eigenvectors = torch.linalg.eigh(covariance)

        # if stochastic, just select one random operation

        num_frames = self.num_frames
        operations = self.operations

        if self.stochastic:
            rand_frame_index = randrange(self.num_frames)

            operations = operations[rand_frame_index:(rand_frame_index + 1)]
            num_frames = 1

        # frames

        frames = rearrange(eigenvectors, 'b d e -> b 1 d e') * rearrange(operations, 'f e -> f 1 e')

        # inverse frame op

        inputs = einsum(frames, centered_points, 'b f d e, b n d -> b f n e')

        # define the frame averaging function

        def frame_average(out):
            if not self.invariant_output:
                # apply frames

                out = einsum(frames, out, 'b f d e, b f ... e -> b f ... d')

            if not self.stochastic:
                # averaging across frames, thus "frame averaging"

                out = reduce(out, 'b f ... -> b ...', 'mean')
            else:
                out = rearrange(out, 'b 1 ... -> b ...')

            return out

        # if one wants to handle the framed inputs externally

        if return_framed_inputs_and_averaging_function or not exists(self.net):

            if self.stochastic and self.return_stochastic_as_augmented_pos:
                return rearrange(inputs, 'b 1 ... -> b ...')

            return inputs, frame_average

        # merge frames into batch

        inputs = rearrange(inputs, 'b f ... -> (b f) ...')

        # if batch is expanded by number of frames, any tensor being passed in for args and kwargs needed to be expanded as well
        # automatically take care of this

        if not self.stochastic:
            args, kwargs = tree_map(
                lambda el: (
                    repeat(el, 'b ... -> (b f) ...', f = num_frames)
                    if torch.is_tensor(el)
                    else el
                )
            , (args, kwargs))

        # main network forward

        out = self.net(inputs, *args, **kwargs)

        # use tree map to handle multiple outputs

        out = tree_map(lambda t: rearrange(t, '(b f) ... -> b f ...', f = num_frames) if torch.is_tensor(t) else t, out)
        out = tree_map(lambda t: frame_average(t) if torch.is_tensor(t) else t, out)

        return out

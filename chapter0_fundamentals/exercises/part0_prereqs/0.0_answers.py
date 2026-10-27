# %%

import math
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"


# %%
arr = np.load(section_dir / "numbers.npy")

# %%
print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch


# %%
print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])  # plotting the first channel of the first image, as monochrome

# %%
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row

# %%
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)

# %%
# Your code here - define arr2
arr2 = einops.repeat(arr[1], 'c h w -> c (repeat h) w', repeat=2)
display_array_as_img(arr2)

# %%
arr3 = einops.repeat(arr[0:2], 'b c h w -> c (b h) w')
display_array_as_img(arr3)

# %%
arr4 = einops.repeat(arr[0], 'c h w -> c (h 2) w')
display_array_as_img(arr4)

# %%
arr5 = einops.repeat(arr[0], 'c h w -> h (c w)')
display_array_as_img(arr5)

# %%
arr6 = einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2)
display_array_as_img(arr6)

# %%
arr7 = einops.rearrange(arr[1], 'c h w -> c w h')
display_array_as_img(arr7)

# %%
arr8 = einops.reduce(arr, '(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)', 'max', b1=2, w2=2, h2=2)
display_array_as_img(arr8)


# %%
# A1

def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    intermediate = t.arange(3, 9)

    result = einops.rearrange(intermediate, '(c1 c2) -> c1 c2', c1 = 3)
    return result

tests.test_rearrange_1(rearrange_1)
# %%
#A2 
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    intermediate = t.arange(1, 7)

    result = einops.rearrange(intermediate, '(c1 c2) -> c1 c2', c1 = 2)
    return result


tests.test_rearrange_2(rearrange_2)
# %%
#B1
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    result = einops.reduce(temps, '(k c) -> k', 'mean', c=7)
    return result


tests.test_temperatures_average(temperatures_average)
# %%
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    print("temps shape ", temps.shape)
    average_temps = einops.reduce(temps, '(k c) -> k', 'mean', c=7)
    print("average temps shape", average_temps.shape)
    average_temps = einops.repeat(average_temps, 'c ->(c 7)')
    print("average temps reshape", average_temps.shape)
    temperatures_differences = temps - average_temps
    return temperatures_differences


tests.test_temperatures_differences(temperatures_differences)
# %%
#B3
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    assert len(temps) % 7 == 0
    average_temps = einops.reduce(temps, '(k c) -> k', 'mean', c=7)
    print("average temps ", average_temps)
    stddev_temps = einops.reduce(temps, '(k c) -> k', t.std, c=7)
    print("stddev ", stddev_temps)
    average_temps = einops.repeat(average_temps, 'c ->(c 7)')
    print("average temps repeated", average_temps)
    temperatures_differences = temps - average_temps
    print("temperatures_differences", temperatures_differences)
    final_temperatures = temperatures_differences / einops.repeat(stddev_temps, 'c ->(c 7)')
    print("final_temperatures", final_temperatures)
    return final_temperatures

tests.test_temperatures_normalized(temperatures_normalized)
# %%
# C1
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    l2n = einops.reduce(matrix, 'm n -> m', lambda x, dims: t.linalg.vector_norm(x, dim=dims))
    print(l2n)
    m, n = matrix.shape
    l2ns = einops.repeat(l2n, 'c -> c k', k = n)
    result = matrix / l2ns
    return result


tests.test_normalize_rows(normalize_rows)
# %%
#C2 

def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    m, n = matrix.shape
    print(m, n)
    normalized = normalize_rows(matrix)
    print(normalized)

    
    return normalized


tests.test_cos_sim_matrix(cos_sim_matrix)

# %%
#D 

def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """

    result = t.ones(n)

    probcum = t.cumsum(probs, 0)
    print(probcum)
    rands = t.rand(n, 1)
    print(rands)
    intermediate = rands > probcum
    print(intermediate)
    out = intermediate.sum(-1) # indexes of first element of probcum greater than rands[i]
    return out


tests.test_sample_distribution(sample_distribution)
# %%
# E
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    print(scores)
    m,n = scores.shape
    guesses = t.argmax(scores, dim=1)
    print("guesses", guesses)
    print(guesses.shape)
    print("true classes", true_classes)
    print(true_classes.shape)
    return (guesses == true_classes).sum(dim=-1)/m
 


tests.test_classifier_accuracy(classifier_accuracy)
# %%
# F1
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """

    print(prices)
    print(items)

    item_prices = t.gather(prices, 0, items)
    print(item_prices)
    return item_prices.sum().item()


tests.test_total_price_indexing(total_price_indexing)

# %%
# F2
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. 
    This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    assert matrix.shape[0] == indexes.shape[0]

    out = matrix.gather(1, indexes)
    # YOUR CODE HERE - add assert statement(s) here for `out`
    assert out[1][0] == matrix[1][indexes[1][0]]
    assert out.shape[0] == matrix.shape[0]
    assert out.shape[1] == indexes.shape[1]

    return out


tests.test_gather_2d(gather_2d)

# %%
# F3

def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    item_prices = t.gather(prices, 0, items)
    print(item_prices)
    return item_prices.sum().item()

tests.test_total_price_gather(total_price_gather)
# %%
# G
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """

    print(matrix)
    print(coords)
    result = matrix[tuple(coords.T)]
    return result


tests.test_integer_array_indexing(integer_array_indexing)
# %%
# H1

def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """

    # https://leimao.github.io/blog/LogSumExp/
    #  Hint: out[i] = log(sum(exp(row i))); reduce over the last dim.
    print(matrix)
    return einops.reduce(matrix, 'i j -> i', logsumexp)

def logsumexp(vector: Tensor, *args) -> float:
    print('vector', vector)
    a = t.log(vector).max()
    print('a', a)
    i1 = vector - a
    print('i1', i1)
    i2 = t.log(i1)
    print('i2', i2)
    i3 = t.exp(i2)
    print('i3', i3)
    return a + i3.sum().item()


tests.test_batched_logsumexp(batched_logsumexp)

# %%

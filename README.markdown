Working Thoughts
====

General Concepts
---

* Integrate predictions over multiple gamma values by scaling each summation term by the normalized SVM risk (or local risk)
* Integrate predicitons over multiple nu/epsilon values by weighting inputs based on their local determinacy
* Consider using a larger optimization problem to produce sparser outputs (diagonal matrix per dimension rather than tensor product)
  Alternately, this could be treated as a post-processing step executed on the SV's


Science!:
---

1. explore risk of hybrid predictor vs. it's constituent predictors
2. Look into weighting points for hybrid nu/epsilon
3. Try per-dimension predictors
4. Look into spatial mapping for scope expansion


Implementation Notes:
---

1. Integrate using risk, integration function (2^-R(x))
2. Shift the decomposition into the GPU using scan()  (is this possible if the kernel matrix won't fit there?)
3. Mask SV's after calculation
4. Cache intermediate results to disk
5. Use the nodes & pipes library so you can use both cores
6. When splitting the kernel matrix, save output to file to free RAM - see if libsvm can read directly from the file


Scale Expansion
---
We can think of scale expansion as either reducing the dimensionality of the event space or of translating that space
into a denser space (equivalent conceptualizations).  The reason this works is that we assume that the density of the
input space isn't uniform - there are certain paths through the space which are more likely than others, which translates
to points in the path-space which are more likely.  If this wasn't the case, we wouldn't be able to make useful
predictions over sequences.  The basic idea of scale expansion is that given this structure over the sequence-space,
we should be able to generate a random variable over the full event space taking values in an alphabet which is a 
subset of the full space's alphabet.  If we consider moving from sequences of n to n^2, the VC dimension of the induced
space will be less than that of the original space, due to this reduced alphabet.

In terms of implementation, we could think of this as a clustering problem; we identify a set of disjoint regions
and assign points in each region to a letter based on the region.  This problem with this approach is that it assumes
disjoint regions, and isn't very elegant.  I think a cleaner (and faster) approach would be to treat the SV's as 
the alphabet (as considered before), and to encode each observation by the closest SV. This gives us access to the
kernel matrix which can be used as a lookup table for distances between observations and eliminates the need to do 
another optimization task (hypersphere SVM) as well as a clustering task (voronoi calculation).  The outputs aren't
as sparse, but they're more reliable, and we can control the sparseness through nu anyway, so it shouldn't be a problem.

There remains a question of how we handle sequences of sequences in the prediction task.  The original idea was to 
split the observations into windows and determine the transition probabilities.  The problem with this task is that 
we end up trying to predict at multiple offsets within the window. Another possibility is to still use windows, but
to slide those windows along the inputs, generating a set of observations for each observation in the base space.
This generates more data, but that's sort of what we want (decreases risk).  This seems like the way to go - gives
us the most robust predictor, even if it means keeping a lot of data.

I guess the data we'd have to keep is the 'winning' SV for each observation. Then at each observation we'd take
the set defined by the endpoint and offset.  It may be possible to translate this into an accumulator model, who knows.

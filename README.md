# Ckmeans

Repeatable unidimensional data clustering inspired by Ckmeans.1d.dp

## Installation

Install the gem and add to the application's Gemfile by executing:

```bash
bundle add ckmeans
```

If bundler is not being used to manage dependencies, install the gem by executing:

```bash
gem install ckmeans
```

## Usage

### Basic Clustering

```rb
# Fixed cluster count (K known in advance)
Ckmeans::Clusterer.new(data, 3).clusters
Ckmedian::Clusterer.new(data, 3).clusters

# Automatic K selection (tries K from kmin to kmax, picks optimal)
Ckmeans::Clusterer.new(data, 1, 10).clusters
Ckmedian::Clusterer.new(data, 1, 10).clusters
```

### Choosing Between Ckmeans and Ckmedian

- **Ckmeans** - Minimizes squared distances (L2). Good for normally distributed data.
- **Ckmedian** - Minimizes absolute distances (L1). More robust to outliers and data bursts.

```rb
# For clean numerical data
temperatures = [20.1, 20.2, 25.5, 25.6, 30.1, 30.2]
Ckmeans::Clusterer.new(temperatures, 1, 5).clusters
# => [[20.1, 20.2], [25.5, 25.6], [30.1, 30.2]]

# For data with outliers (e.g., photo timestamps with bursts)
timestamps = photos.map(&:taken_at).map(&:to_i)
Ckmedian::Clusterer.new(timestamps, 1, 20).clusters
```

### Stable Estimation (Recommended for Edge Cases)

By default, both algorithms use a fast heuristic for estimating K. For datasets with many duplicates, tight clusters, or outliers, use `:stable` for more robust estimation:

```rb
# Stable estimation (uses statistical mixture models)
Ckmeans::Clusterer.new(data, 1, 10, :stable).clusters
Ckmedian::Clusterer.new(data, 1, 10, :stable).clusters
```

**When to use `:stable`:**
- Small to medium datasets (< 1000 points)
- Many duplicate values
- Clusters with very different sizes
- Photo/event timeline clustering (bursts and gaps)

**Expert users:** `:stable` is an alias for `:gmm` (Gaussian Mixture Model) in Ckmeans and `:lmm` (Laplace Mixture Model) in Ckmedian.

## License

The gem is available as open source under the terms of the [LGPL v3 License](https://opensource.org/license/lgpl-3-0).

## References

- https://github.com/cran/Ckmeans.1d.dp

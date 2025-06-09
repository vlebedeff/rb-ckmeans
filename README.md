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

### Fixed Cluster Count

```rb
# Fixed cluster count
Ckmeans::Clusterer(data, kmin).clusters
Ckmedian::Clusterer(data, kmin).clusters
```

### Estimate optimal cluster count within kmin and kmax

```rb
Ckmeans::Clusterer(data, kmin, kmax).clusters
Ckmedian::Clusterer(data, kmin, kmax).clusters
```

### Fast & Stable Estimation of K

For big collections without many duplicates, use regular estimation.
For relatively small sets or sets with many duplicates use Gaussian Mixture Model (GMM)-based estimation.
It works slower but is more resilient for various data patterns like big numbers of duplicates or clusters with different
numbers of elements.

```rb
Ckmeans::Clusterer(data, kmin, kmax, :gmm).clusters
Ckmedian::Clusterer(data, kmin, kmax, :gmm).clusters
```

## License

The gem is available as open source under the terms of the [LGPL v3 License](https://opensource.org/license/lgpl-3-0).

## References

- https://github.com/cran/Ckmeans.1d.dp

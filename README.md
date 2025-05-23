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

```rb
# Fixed cluster count
Ckmeans::Clusterer(data, kmin).clusters
Ckmedian::Clusterer(data, kmin).clusters

# Estimate optimal cluster count within kmin and kmax
Ckmeans::Clusterer(data, kmin, kmax).clusters
Ckmedian::Clusterer(data, kmin, kmax).clusters

# Adjust Bayesian Information Criteria favoring more smaller clusters (Ckmeans only)
Ckmeans::Clusterer(data, kmin, kmax, :sensitive).clusters
```

## License

The gem is available as open source under the terms of the [LGPL v3 License](https://opensource.org/license/lgpl-3-0).

## References

- https://github.com/cran/Ckmeans.1d.dp

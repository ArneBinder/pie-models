# pie-models

Model and Taskmodule implementations for [PyTorch-IE](https://github.com/ChristophAlt/pytorch-ie).

## Setup

```bash
pip install git+https://github.com/ArneBinder/pie-models.git
```

or

```bash
pip install git+ssh://git@github.com/ArneBinder/pie-models.git
```

## Release Process

1. create a branch `release` from the `main` branch
2. bump the version in `setup.py`. If the release contains new features, or breaking changes, bump the minor version (this project has not yet any main release). If the release contains only bugfixes, bump the patch version. See [Semantic Versioning](https://semver.org/) for more information.
3. commit and push the changes
4. create a pull request from `release` to `main`
5. wait for the CI to pass
6. merge the pull request and delete `release` branch (this is important, because otherwise the next release will fail)
7. create a new release on GitHub via the "Releases" tab and click on "Draft a new release".
   1. Click on "Choose a tag" and create a new one which should be the same as the version in `setup.py`, but prefixed with `v`, e.g. `v0.6.1` for version `0.6.1`.
   2. You can choose an appropriate release title.
   3. Click on "Generate release notes" to generate the release notes from the pull request descriptions.
   4. When everything looks fine, click on "Publish release" to publish the release.

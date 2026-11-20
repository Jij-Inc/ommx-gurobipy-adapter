# Releasing the adapter

The adapter version is derived from Git tags by `setuptools_scm`; it is
independent of the OMMX SDK version. Use a PEP 440 tag such as `v0.3.0rc1`
for an adapter release candidate and `v0.3.0` for its stable release.

Before tagging, merge the release changes and require passing Python 3.11–3.14
tests and the Release build job. The latter builds and checks both wheel and
sdist, installs the wheel in isolation and solves a small model, and rebuilds
the sdist into a wheel. Pull requests and manual runs only build artifacts.
Pushing a `v*` tag publishes the distributions to PyPI using the repository's
existing Trusted Publishing configuration. Verify that configuration before
the first release. Tag pushes publish automatically, so choose the version
before pushing the tag.

## OMMX RC1 to 3.0.0

The runtime requirement `ommx>=3.0.0rc1,<4.0.0` accepts RC1 and the final
3.0.0 release without changing or republishing the adapter. The lockfile pins
RC1 for reproducible development and CI; it does not pin consumers of the wheel.
Users can upgrade to the final SDK with `pip install --upgrade ommx` once it is
available. Compatibility with the final release must still be verified when
that release is published.

For the stable adapter release, update the lockfile with
`uv lock --upgrade-package ommx`, confirm it resolves to the released 3.0.0
(or the intended newer 3.x), run the test and distribution build jobs, and tag
the stable adapter version. If support for prerelease SDKs should end, raise
the lower bound to `ommx>=3.0.0,<4.0.0` in `pyproject.toml` and README before
regenerating the lockfile. No adapter implementation change is expected if
the final SDK preserves the RC1 API.

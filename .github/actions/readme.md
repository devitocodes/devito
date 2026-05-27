# Devito Actions

The Devito actions can either be used from the Github interface:
```yaml
uses: devitocodes/devito/.github/actions/docker-build@main
```

Or internally from within the repository:
```yaml
uses: ./.github/actions/docker-run
```

## `docker-build`

Inputs:

- `file`: Dockerfile containing build instructions (default: `Dockerfile`)
- `tag`: Tag to add to the built image
- `base`: Base docker image to build on top of
- `args`: Arguments to pass to `docker build`

Outputs:

- `unique`: Unique identifier for the CI run (ie: `${{ steps.build.outputs.unique }}`)

Example:

```yaml
jobs:
  test:
    steps:
    - id: build
      name: Build docker image for Devito
      uses: ./.github/actions/docker-build
      with:
        file: docker/Dockerfile.devito
        tag: tag-related-to-test-config
        base: base-image-to-build-from
        args: "--more-docker-build-args"
```

## `docker-run`

Inputs:

- `uid`: Unique identifier output from docker-build action
- `tag`: Tag of the built image to use
- `name`: Name substring for docker to use when running the command (optional)
- `args`: Arguments to pass to `docker run`, `--init -t --rm` are always added
- `env`: Environment variables to set inside the docker container, one environment variable per line
-command: Command to execute inside of the docker container

### Notes

- The UID must be unique, easily obtained from build action
- The tag must match built image, easily obtained from build action
- If you provide a custom name `foo` the container name will be `ci-foo-UUUUUUUUUU` where UUUUUUUUUU is the UID
- The default args `--init -t --rm` are _always_ added
- Environment variables must be passed a single environment variable per line, best achieved with the (`|`) syntax in yaml
- Only a single command is executed, not a list of commands. Using `;` or `&&` will result in subsequent commands being executed outside of the docker environment

Example:

```yaml
jobs:
  test:
    steps:
    - name: Run a command in a previously built Docker container
      uses: ./.github/actions/docker-run
      with:
        uid: ${{ steps.build.outputs.unique }}
        tag: tag-related-to-test-config
        name: completely-optional-name
        args: "--more-docker-run-args"
        env: |
          FOO=value1
          BAR=value2
        command: |
          mpiexec -n 4 \
            python \
              my_complicated_script.py --arg1 -v
```

## `docker-clean`

Inputs:

- `uid`: Unique identifier output from docker-build action
- `tag`: Tag of the built image to use

### Notes

- UID must be unique, easily obtained from build action
- Tag must match built image, easily obtained from build action
- Use `if: always()` to always clean up the image, even if the workflow fails

Example:

```yaml
jobs:
  test:
    steps:
    - name: Cleanup Docker image
      if: always()
      uses: ./.github/actions/docker-clean
      with:
        uid: ${{ steps.build.outputs.unique }}
        tag: tag-related-to-test-config
```

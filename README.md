# gotorch

We use docker as the development environment. To build the image, run the following command

```bash
docker build -t gotorch:dev .
```

To run the example, run the following command

```bash
docker run --rm -it -v $PWD:/work -w /work gotorch:dev ./build_and_test.sh
```

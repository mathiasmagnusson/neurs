use neurs::*;
use std::fs;

fn main() {
    let input = String::from_utf8(fs::read("neuralnetwork-layers.json").unwrap()).unwrap();

    fn sigmoid(x: Scalar) -> Scalar { x * 2.0 };

    let network = Network::builder()
        .parse_json_source(&input)
        .unwrap()
        .with_sigmoid_fn(SigmoidFunction::Custom(sigmoid))
        .build();

    for f in 0.. {
        let output = network.calc(&[f as Scalar * 2.0, -f as Scalar, f as Scalar * -3.0]);

        println!(
            "[{: <8}, {: <8}, {: <8}] => {}",
            f * 2,
            -f,
            -f * 3,
            output[0]
        );
    }
}

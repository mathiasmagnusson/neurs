pub type Scalar = f32;

pub enum SigmoidFunction {
	XOver1PlusAbsOfX,
	Custom(fn(Scalar) -> Scalar),
}

impl SigmoidFunction {
	pub fn calculate(&self, x: Scalar) -> Scalar {
		use self::SigmoidFunction::*;
		match self {
			XOver1PlusAbsOfX => x / (1.0 + x.abs()),
			Custom(f) => f(x),
		}
	}
}

pub struct NetworkBuilder {
	layers: Vec<Layer>,
	sigmoid_fn: SigmoidFunction,
}

impl NetworkBuilder {
	pub fn layer<L: Into<Layer>>(mut self, layer: L) -> Self {
		self.layers.push(layer.into());
		self
	}

	pub fn layers(mut self, mut layers: Vec<Layer>) -> Self {
		self.layers.append(&mut layers);
		self
	}

	pub fn with_sigmoid_fn(mut self, sigmoid_fn: SigmoidFunction) -> Self {
		self.sigmoid_fn = sigmoid_fn;
		self
	}

	pub fn parse_json_source(self, source: &str) -> Result<Self, ()> {
		let json = json::parse(source);
		if let Ok(json) = json {
			self.parse_json(json)
		} else {
			Err(())
		}
	}

	pub fn parse_json(mut self, json: json::JsonValue) -> Result<Self, ()> {
		if !json.is_array() {
			return Err(());
		}

		let mut layers: Vec<Layer> = vec![];

		for json_layer in json.members() {
			if !json_layer.is_array() {
				return Err(());
			}

			let mut neurons: Vec<Neuron> = vec![];

			for neuron in json_layer.members() {
				let mut weights: Vec<Scalar> = vec![];

				let json_weights = &neuron["weights"];
				if !json_weights.is_array() {
					return Err(());
				}

				for weight in json_weights.members() {
					if let Some(weight) = weight.as_f64() {
						weights.push(weight as Scalar);
					} else {
						return Err(());
					}
				}

				let bias: Scalar = if let Some(bias) = neuron["bias"].as_f64() {
					bias as Scalar
				} else {
					return Err(());
				};

				neurons.push(Neuron::new(weights, bias));
			}

			layers.push(Layer::new(neurons));
		}

		self.layers = layers;

		Ok(self)
	}

	pub fn build(self) -> Network {
		Network::new(self.layers, self.sigmoid_fn)
	}
}

pub struct Network {
	layers: Vec<Layer>,
	sigmoid_fn: SigmoidFunction,
}

impl Network {
	pub fn builder() -> NetworkBuilder {
		NetworkBuilder {
			layers: vec![],
			sigmoid_fn: SigmoidFunction::XOver1PlusAbsOfX,
		}
	}

	pub fn new(layers: Vec<Layer>, sigmoid_fn: SigmoidFunction) -> Self {
		Self { layers, sigmoid_fn }
	}

	pub fn calc(&self, input: &[Scalar]) -> Vec<Scalar> {
		// The output vector from the last layer
		let mut llo = input.to_vec();

		for layer in self.layers.iter() {
			llo = layer.calc(&llo, &self.sigmoid_fn);
		}

		llo
	}
}

pub struct Layer {
	neurons: Vec<Neuron>,
}

impl Into<Layer> for Vec<Neuron> {
	fn into(self) -> Layer {
		Layer::new(self)
	}
}

impl Into<Layer> for Vec<(Vec<Scalar>, Scalar)> {
	fn into(self) -> Layer {
		Layer::new(self.into_iter().map(|n| Neuron::new(n.0, n.1)).collect())
	}
}

impl Layer {
	pub fn new(neurons: Vec<Neuron>) -> Self {
		Self { neurons }
	}

	pub fn calc(&self, input: &[Scalar], sigmoid_fn: &SigmoidFunction) -> Vec<Scalar> {
		self.neurons
			.iter()
			.map(|neuron| neuron.calc(input, sigmoid_fn))
			.collect()
	}
}

pub struct Neuron {
	weights: Vec<Scalar>,
	bias: Scalar,
}

impl Neuron {
	pub fn new(weights: Vec<Scalar>, bias: Scalar) -> Self {
		Self { weights, bias }
	}

	pub fn calc(&self, input: &[Scalar], sigmoid_fn: &SigmoidFunction) -> Scalar {
		debug_assert_eq!(input.len(), self.weights.len());

		sigmoid_fn.calculate(
			input
				.iter()
				.zip(self.weights.iter())
				.map(|(input, weight)| input * weight)
				.sum::<Scalar>()
				+ self.bias,
		)
	}
}

use anyhow::{anyhow, Result};
use rand::{distributions::Uniform, prelude::ThreadRng, Rng};
use std::collections::HashMap;
fn main() {
    let study = create_study(Storage::new(), Sampler::new());
    study.optimize(obj, 10);
}
fn obj(trial: &mut Trial) -> f64 {
    let x = trial.suggest_int("x", 0, 10).unwrap();
    let y = trial.suggest_int("y", 0, 10).unwrap();
    return (x as f64 - 3_f64).powf(2.0) + (y as f64 - 5_f64).powf(2.0);
}

trait Distribution<T> {
    fn to_internal_repr(&self, external_repr: T) -> f64;
    fn to_external_repr(&self, internal_repr: f64) -> T;
}
#[derive(Clone)]
struct IntUniformDistribution {
    low: i64,
    high: i64,
}
impl IntUniformDistribution {
    fn new(low: i64, high: i64) -> Self {
        IntUniformDistribution { low, high }
    }
}
impl Distribution<i64> for IntUniformDistribution {
    fn to_internal_repr(&self, external_repr: i64) -> f64 {
        return external_repr as f64;
    }

    fn to_external_repr(&self, internal_repr: f64) -> i64 {
        return internal_repr as i64;
    }
}
#[derive(Clone, Debug)]
struct UniformDistribution {
    low: f64,
    high: f64,
}
impl UniformDistribution {
    fn new(low: f64, high: f64) -> Self {
        UniformDistribution { low, high }
    }
}
impl Distribution<f64> for UniformDistribution {
    fn to_internal_repr(&self, external_repr: f64) -> f64 {
        return external_repr;
    }

    fn to_external_repr(&self, internal_repr: f64) -> f64 {
        return internal_repr;
    }
}
#[derive(Clone, Debug)]
struct LogUniformDistribution {
    low: f64,
    high: f64,
}
impl LogUniformDistribution {
    fn new(low: f64, high: f64) -> Self {
        LogUniformDistribution { low, high }
    }
}
impl Distribution<f64> for LogUniformDistribution {
    fn to_internal_repr(&self, external_repr: f64) -> f64 {
        external_repr
    }

    fn to_external_repr(&self, internal_repr: f64) -> f64 {
        internal_repr
    }
}

#[derive(Clone)]
struct CategoricalDistribution {
    choices: Vec<String>,
}
impl CategoricalDistribution {
    fn new(choices: Vec<String>) -> Self {
        CategoricalDistribution { choices }
    }
}

impl Distribution<String> for CategoricalDistribution {
    fn to_internal_repr(&self, external_repr: String) -> f64 {
        return self
            .choices
            .iter()
            .position(|choice| *choice == external_repr)
            .unwrap_or(0) as f64;
    }

    fn to_external_repr(&self, internal_repr: f64) -> String {
        return self.choices[internal_repr as usize].clone();
    }
}

/// see https://www.simonewebdesign.it/rust-hashmap-insert-values-multiple-types/
#[derive(Clone)]
enum Distributions {
    Uni(UniformDistribution),
    IntUni(IntUniformDistribution),
    Categorical(CategoricalDistribution),
    LogUni(LogUniformDistribution),
}

enum ExternalRepr {
    Int(i64),
    Float(f64),
    Str(String),
}
#[derive(PartialEq, Clone, Copy)]
enum FrozenTrialState {
    Running,
    Completed,
    Failed,
}

#[derive(Clone)]
struct FrozenTrial {
    trial_id: usize,
    state: FrozenTrialState,
    value: f64,
    internal_params: HashMap<String, f64>,
    distributions: HashMap<String, Distributions>,
}

impl FrozenTrial {
    fn new(trial_id: usize, state: FrozenTrialState, value: f64) -> Self {
        Self {
            trial_id,
            state,
            value,
            internal_params: HashMap::new(),
            distributions: HashMap::new(),
        }
    }

    fn is_finised(&self) -> bool {
        match self.state {
            FrozenTrialState::Running => false,
            _ => true,
        }
    }

    fn params(&mut self) -> HashMap<String, ExternalRepr> {
        let mut external_repr: HashMap<String, ExternalRepr> = HashMap::new();
        for param_name in self.internal_params.keys() {
            let distribution = self.distributions.get_mut(param_name).unwrap();
            let internal_repr = self.internal_params[param_name];
            match distribution {
                Distributions::Uni(dist) => {
                    external_repr.insert(
                        param_name.to_string(),
                        ExternalRepr::Float(dist.to_external_repr(internal_repr)),
                    );
                }
                Distributions::IntUni(dist) => {
                    external_repr.insert(
                        param_name.to_string(),
                        ExternalRepr::Int(dist.to_external_repr(internal_repr)),
                    );
                }
                Distributions::Categorical(dist) => {
                    external_repr.insert(
                        param_name.to_string(),
                        ExternalRepr::Str(dist.to_external_repr(internal_repr)),
                    );
                }
                Distributions::LogUni(dist) => {
                    external_repr.insert(
                        param_name.to_string(),
                        ExternalRepr::Float(dist.to_external_repr(internal_repr)),
                    );
                }
            };
        }
        return external_repr;
    }
}

#[derive(Clone)]
struct Storage {
    trials: Vec<FrozenTrial>,
}
impl Storage {
    fn new() -> Self {
        Storage { trials: vec![] }
    }

    fn create_new_trial(&mut self) -> usize {
        let trial_id = self.trials.len();
        let params: HashMap<String, f64> = HashMap::new();
        let trial = FrozenTrial::new(trial_id, FrozenTrialState::Running, 0_f64);
        self.trials.push(trial);
        return trial_id;
    }

    fn get_trial(&self, trial_id: usize) -> Result<FrozenTrial> {
        let target = self
            .trials
            .iter()
            .filter(|&trial| trial.trial_id == trial_id)
            .collect::<Vec<&FrozenTrial>>();

        if let Some(&res) = target.first() {
            return Ok(res.clone());
        } else {
            return Err(anyhow!("Missing trial id: {}", trial_id));
        }
    }

    fn get_best_trial(&self) -> Option<FrozenTrial> {
        let mut completed_trials: Vec<&FrozenTrial> = self
            .trials
            .iter()
            .filter(|&trial| trial.state == FrozenTrialState::Completed)
            .filter(|&trial| trial.value.is_finite())
            .collect();
        completed_trials.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap());
        if let Some(&res) = completed_trials.first() {
            return Some(res.clone());
        } else {
            return None;
        }
    }

    fn set_trial_value(&mut self, trial_id: usize, value: f64) -> Result<()> {
        let mut target_idx = -1;
        for i in 0..self.trials.len() {
            let trial = &self.trials[i];
            if trial.trial_id == trial_id {
                if trial.is_finised() {
                    return Err(anyhow!("Cannot update finished tirals"));
                }
                target_idx = i as i64;
            }
        }

        if target_idx < 0 {
            return Err(anyhow!("Missing trial idx: {}", trial_id));
        }
        self.trials[target_idx as usize].value = value;
        return Ok(());
    }

    fn set_trial_state(&mut self, trial_id: usize, state: FrozenTrialState) -> Result<()> {
        let mut target_idx = -1;
        for i in 0..self.trials.len() {
            let trial = &self.trials[i];
            if trial.trial_id == trial_id {
                if trial.is_finised() {
                    return Err(anyhow!("Cannot update finished tirals"));
                }
                target_idx = i as i64;
            }
        }

        if target_idx < 0 {
            return Err(anyhow!("Missing trial idx: {}", trial_id));
        }
        self.trials[target_idx as usize].state = state;
        return Ok(());
    }

    fn set_trial_param(
        &mut self,
        trial_id: usize,
        name: &str,
        distribution: Distributions,
        value: f64,
    ) -> Result<()> {
        let mut target_idx = -1;
        for i in 0..self.trials.len() {
            let trial = &self.trials[i];
            if trial.trial_id == trial_id {
                if trial.is_finised() {
                    return Err(anyhow!("Cannot update finished tirals"));
                }
                target_idx = i as i64;
            }
        }

        if target_idx < 0 {
            return Err(anyhow!("Missing trial idx: {}", trial_id));
        }
        self.trials[target_idx as usize]
            .internal_params
            .insert(name.to_string(), value);
        self.trials[target_idx as usize]
            .distributions
            .insert(name.to_string(), distribution);
        return Ok(());
    }
}

struct Trial {
    study: Study,
    trial_id: usize,
}

impl Trial {
    fn new(study: Study, trial_id: usize) -> Self {
        return Trial {
            study: study,
            trial_id: trial_id,
        };
    }

    fn suggest_uniform(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        let trial = self.study.storage.get_trial(self.trial_id);
        let distribution = UniformDistribution::new(low, high);
        let distributionEnum = Distributions::Uni(UniformDistribution::new(low, high));
        let param_value = self
            .study
            .sampler
            .sample_independent(name, distributionEnum);
        let param_value_in_internal_repr = distribution.to_internal_repr(param_value);
        self.study.storage.set_trial_param(
            self.trial_id,
            name,
            Distributions::Uni(distribution),
            param_value_in_internal_repr,
        );
        return Ok(param_value);
    }

    fn suggest_log(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        let trial = self.study.storage.get_trial(self.trial_id);
        let distribution = LogUniformDistribution::new(low, high);
        let distributionEnum = Distributions::LogUni(LogUniformDistribution::new(low, high));
        let param_value = self
            .study
            .sampler
            .sample_independent(name, distributionEnum);
        let param_value_in_internal_repr = distribution.to_internal_repr(param_value);
        self.study.storage.set_trial_param(
            self.trial_id,
            name,
            Distributions::LogUni(distribution),
            param_value_in_internal_repr,
        );
        return Ok(param_value);
    }

    fn suggest_categorical(&mut self, name: &str, choices: Vec<String>) -> Result<String> {
        let trial = self.study.storage.get_trial(self.trial_id);
        let distribution = CategoricalDistribution::new(choices.clone());
        let distributionEnum = Distributions::Categorical(CategoricalDistribution::new(choices));
        let param_value = self
            .study
            .sampler
            .sample_independent_category(name, distributionEnum);
        let param_value_in_internal_repr = distribution.to_internal_repr(param_value.clone());
        self.study.storage.set_trial_param(
            self.trial_id,
            name,
            Distributions::Categorical(distribution),
            param_value_in_internal_repr,
        );
        return Ok(param_value);
    }

    fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        let trial = self.study.storage.get_trial(self.trial_id);
        let distribution = IntUniformDistribution::new(low, high);
        let distributionEnum = Distributions::IntUni(IntUniformDistribution::new(low, high));
        let param_value = self
            .study
            .sampler
            .sample_independent_int(name, distributionEnum);
        let param_value_in_internal_repr = distribution.to_internal_repr(param_value);
        self.study.storage.set_trial_param(
            self.trial_id,
            name,
            Distributions::IntUni(distribution),
            param_value_in_internal_repr,
        );
        return Ok(param_value);
    }
}

#[derive(Clone)]
struct Sampler {
    rng: ThreadRng,
}

impl Sampler {
    fn new() -> Self {
        Sampler {
            rng: rand::thread_rng(),
        }
    }

    fn sample_independent(&mut self, name: &str, distribution: Distributions) -> f64 {
        match distribution {
            Distributions::Uni(dist) => {
                let dice = rand::distributions::Uniform::from(dist.low..=dist.high);
                let n = self.rng.sample(dice);
                return n;
            }
            Distributions::LogUni(dist) => {
                let log_low = dist.low.ln();
                let log_high = dist.high.ln();
                let dice = Uniform::from(log_low..=log_high);
                let n = self.rng.sample(dice);
                return n.exp();
            }
            _ => {
                return 0.0;
            }
        }
    }

    fn sample_independent_int(&mut self, name: &str, distribution: Distributions) -> i64 {
        match distribution {
            Distributions::IntUni(dist) => {
                let dice = Uniform::from(dist.low..=dist.high);
                let n = self.rng.sample(dice);
                return n;
            }
            _ => {
                return 0;
            }
        }
    }

    fn sample_independent_category(&mut self, name: &str, distribution: Distributions) -> String {
        match distribution {
            Distributions::Categorical(dist) => {
                let idx = self.rng.gen_range(0..dist.choices.len());
                return dist.choices[idx].clone();
            }
            _ => "".to_string(),
        }
    }
}
type Objective = fn(&mut Trial) -> f64;
#[derive(Clone)]
struct Study {
    storage: Storage,
    sampler: Sampler,
}

impl Study {
    fn new(storage: Storage, sampler: Sampler) -> Self {
        Study {
            storage: storage,
            sampler: sampler,
        }
    }

    fn optimize(mut self, objective: Objective, n_trials: u64) {
        for _ in 0..n_trials {
            let trial_id = self.storage.create_new_trial();
            let mut trial = Trial::new(self.clone(), trial_id);
            let value = objective(&mut trial);
            println!("trial_id={} is completed with valud={}", trial_id, value);
            self.storage.set_trial_value(trial_id, value);
            self.storage
                .set_trial_state(trial_id, FrozenTrialState::Completed);
        }
    }

    fn best_trial(&self) -> Option<FrozenTrial> {
        self.storage.get_best_trial()
    }
}

fn create_study(storage: Storage, sampler: Sampler) -> Study {
    return Study {
        storage: storage,
        sampler: sampler,
    };
}

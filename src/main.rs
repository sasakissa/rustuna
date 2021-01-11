use anyhow::{anyhow, Result};
use rand::{prelude::ThreadRng, Rng};
use std::collections::HashMap;
fn main() {
    let study = create_study(Storage::new(), Sampler::new());
    study.optimize(obj, 10);
}
fn obj(trial: &mut Trial) -> f64 {
    let x = trial.suggest_uniform("x", 0.0, 10.0).unwrap();
    let y = trial.suggest_uniform("y", 0.0, 10.0).unwrap();
    return (x - 3_f64).powf(2.0) + (y - 5_f64).powf(2.0);
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
    params: HashMap<String, f64>,
}

impl FrozenTrial {
    fn new(
        trial_id: usize,
        state: FrozenTrialState,
        value: f64,
        params: HashMap<String, f64>,
    ) -> Self {
        Self {
            trial_id,
            state,
            value,
            params,
        }
    }

    fn is_finised(&self) -> bool {
        match self.state {
            FrozenTrialState::Running => false,
            _ => true,
        }
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
        let trial = FrozenTrial::new(trial_id, FrozenTrialState::Running, 0_f64, params);
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

    fn set_trial_param(&mut self, trial_id: usize, name: &str, value: f64) -> Result<()> {
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
            .params
            .insert(name.to_string(), value);
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
        let param = self.study.sampler.sample_independent(low, high);

        self.study
            .storage
            .set_trial_param(self.trial_id, name, param);
        return Ok(param);
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
    fn sample_independent(&mut self, low: f64, high: f64) -> f64 {
        let n: f64 = self.rng.gen_range(low..=high);
        n
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

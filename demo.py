# demo.py
import os

import jax
import jax.numpy as jnp
import numpy as np

from axlearn.common.inference import InferenceRunner
from axlearn.experiments import get_named_trainer_config

def get_runner(name, checkpoint_path):
    """Make an inference runner initialized with pre-trained state according to name."""
    trainer_cfg = get_named_trainer_config(
            name,
            config_module=f"axlearn.experiments.text.gpt.c4_trainer",
        )()
    inference_runner_cfg = InferenceRunner.default_config().set(
        name=f"{name}_inference_runner",
        mesh_axis_names=("data", "expert", "fsdp", "seq", "model"),
        mesh_shape=(1, 1, len(jax.devices()), 1, 1),
        model=trainer_cfg.model.set(dtype=jnp.bfloat16),
        inference_dtype=jnp.bfloat16,
    )
    inference_runner_cfg.init_state_builder.dir = checkpoint_path
    inference_runner = inference_runner_cfg.instantiate(parent=None)
    return inference_runner

def predict(inference_runner, inputs_ids):
    """
    Helper method to perform one forward pass for the model.
    """

    input_batches = [{"input_ids": jnp.array(inputs_ids)}]
    for result in inference_runner.run(
        input_batches,
        method="predict",
        prng_key=jax.random.PRNGKey(11),
    ):
        return result["outputs"]

def gen_tokens(inference_runner, inputs_ids, max_new_tokens):
    """
    Helper method to generate multiple tokens for the model.
    """
    batch_size, prompt_len = inputs_ids.shape

    result_len = prompt_len + max_new_tokens
    result = np.zeros(
        (batch_size, result_len), dtype=inputs_ids.dtype)
    result[:, :prompt_len] = inputs_ids

    input_batches = [{"input_ids": jnp.array(inputs_ids),
    "prefix": jnp.array(result)}]

    for result in inference_runner.run(
        input_batches,
        method="sample_decode",
        prng_key=jax.random.PRNGKey(11),
    ):
        return result["outputs"]

def get_data(seq_len, vocab_size, batch_size=1):
    """ Generate random input in shape of [batch, seq] """
    rng = np.random.RandomState(11)
    input_ids = rng.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
    return input_ids

if __name__ == "__main__":
    model_name = 'fuji-7B-v1-single-host'
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    vocab_size = 32768
    seq_len = 10
    
    model = get_runner(
            model_name,
            checkpoint_path=checkpoint_path,
        )

    input_data = get_data(seq_len, vocab_size)
    print(f'Creating random input ids: {input_data}')

    print("Extracting logits after 1 step.....")
    predict_output = predict(model, input_data)
    logits = predict_output['logits']
    res = np.array(logits).astype(np.float32)
    last_logit = logits[:,-1,:]
    print(f'Last logits are {last_logit}')
    token_id = np.argmax(res[:,-1,:], axis=-1)
    print(f'Predicated token is {token_id}')
    
    max_new_tokens = 5
    print(f'Generating {max_new_tokens} extended tokens.....', )
    decoding_output = gen_tokens(model, input_data, max_new_tokens)
    new_tokens = decoding_output.sequences[:,:, 10:]
    print(f'Extend tokens are {new_tokens}',)
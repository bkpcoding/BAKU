import os
import h5py
import pickle as pkl
import numpy as np
from pathlib import Path

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("/nfs/turbo/coe-mandmlab/shared_data/sagar_corl2025/pkl_libero/")
BENCHMARKS = ["libero_10", "libero_90"]
SAVE_DATA_PATH = Path("../../expert_demos/libero_segmented") # Changed save path
img_size = (128, 128)

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# benchmark for suite
benchmark_dict = benchmark.get_benchmark_dict()

# load sentence transformer
lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load skill segmentation file
LABELED_FILE = "/home/bpatil/workspace/skill_seg/skill_seg/data/skill_dataset1.hdf5" # Path to your segmentation file

def get_demo_skill_segments(labeled_file_path):
    demo_skill_segments = {}
    with h5py.File(labeled_file_path, 'r') as f:
        for demo_key in f:
            demo_group = f[demo_key]
            segment_keys = sorted(demo_group.keys(), key=lambda x: int(x.split('_')[1])) # Sort segment keys
            skills = []
            start_indices = []
            end_indices = []
            task_instructions = [] # List to store task instructions for each segment
            for segment_key in segment_keys:
                segment = demo_group[segment_key]
                skills.append(segment.attrs['skill'])
                start_indices.append(segment.attrs['start_idx']) # Corrected attribute name to 'start_idx'
                end_indices.append(segment.attrs['end_idx'])     # Corrected attribute name to 'end_idx'
                task_instructions.append(segment.attrs['language_instruction']) # Get task instruction for each segment
            # Get task_instruction from the first segment (assuming it's the same for all segments in the demo_group)
            task_instruction = task_instructions[0] # Still using the first task instruction as the overall task instruction for the demo group.
            demo_skill_segments[task_instruction] = {
                demo_key: {'skills': skills, 'start_indices': start_indices, 'end_indices': end_indices, 'task_instructions': task_instructions} # Added task_instructions to the output
            }
    return demo_skill_segments

demo_segments_map = get_demo_skill_segments(LABELED_FILE)

# Total number of tasks
num_tasks = 0
for benchmark in BENCHMARKS:
    benchmark_path = DATASET_PATH / benchmark
    num_tasks += len(list(benchmark_path.glob("*.hdf5")))

tasks_stored = 0
for benchmark in BENCHMARKS:
    print(f"############################# {benchmark} #############################")
    benchmark_path = DATASET_PATH / benchmark

    save_benchmark_path = SAVE_DATA_PATH / benchmark
    save_benchmark_path.mkdir(parents=True, exist_ok=True)

    # Init env benchmark suite
    task_suite = benchmark_dict[benchmark]()
    for task_file in benchmark_path.glob("*.hdf5"):
        print(f"Processing {tasks_stored+1}/{num_tasks}: {task_file}")
        data = h5py.File(task_file, "r")["data"]
        task_name_from_file = str(task_file).split("/")[-1][:-5] # Extract task name from file name

        # Init env - moved outside demo loop to avoid re-initialization
        task_name = str(task_file).split("/")[-1][:-10] # Original task name extraction - might need adjustment
        task_id = task_suite.get_task_names().index(task_name)
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": img_size[0],
            "camera_widths": img_size[1],
        }
        env = OffScreenRenderEnv(**env_args)


        for demo in data.keys():
            print(f"Processing demo: {demo}")
            demo_data = data[demo]
            full_demo_states = np.array(demo_data["states"], dtype=np.float32)
            full_demo_actions = np.array(demo_data["actions"], dtype=np.float32)
            full_demo_rewards = np.array(demo_data["rewards"], dtype=np.float32)

            if task_name_from_file in demo_segments_map and demo in demo_segments_map[task_name_from_file]:
                skill_segment_info = demo_segments_map[task_name_from_file][demo]
                skills = skill_segment_info['skills']
                start_indices = skill_segment_info['start_indices']
                end_indices = skill_segment_info['end_indices']
                task_instructions = skill_segment_info['task_instructions'] # Retrieve task_instructions for each segment

                segmented_observations = {}
                segmented_states = {}
                segmented_actions = {}
                segmented_rewards = {}

                for skill_idx in range(len(skills)):
                    skill_name = skills[skill_idx]
                    start_idx = start_indices[skill_idx]
                    end_idx = end_indices[skill_idx]
                    task_instruction = task_instructions[skill_idx] # Get task instruction for the current skill

                    if start_idx >= len(full_demo_states) or end_idx >= len(full_demo_states):
                        print(f"Warning: Indices out of bounds for demo {demo}, skill {skill_name}. Skipping segment.")
                        continue

                    # Re-init env for each skill with its specific task instruction
                    env_args = {
                        "bddl_file_name": task_bddl_file, # task_bddl_file is still from the overall task, might need adjustment if task_instruction changes the bddl file.
                        "camera_heights": img_size[0],
                        "camera_widths": img_size[1],
                        "language_instruction": task_instruction # Set language instruction for the current skill
                    }
                    env = OffScreenRenderEnv(**env_args)

                    states_segment = full_demo_states[start_idx:end_idx+1]
                    actions_segment = full_demo_actions[start_idx:end_idx+1]
                    rewards_segment = full_demo_rewards[start_idx:end_idx+1]

                    pixels, pixels_ego = [], []
                    joint_states, eef_states, gripper_states = [], [], []
                    for i in range(len(states_segment)):
                        obs = env.regenerate_obs_from_state(states_segment[i])
                        img = obs["agentview_image"][::-1]
                        img_ego = obs["robot0_eye_in_hand_image"][::-1]
                        joint_state = obs["robot0_joint_pos"]
                        eef_state = np.concatenate(
                            [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                        )
                        gripper_state = obs["robot0_gripper_qpos"]
                        # append
                        pixels.append(img)
                        pixels_ego.append(img_ego)
                        joint_states.append(joint_state)
                        eef_states.append(eef_state)
                        gripper_states.append(gripper_state)

                    observation = {}
                    observation["pixels"] = np.array(pixels, dtype=np.uint8)
                    observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
                    observation["joint_states"] = np.array(joint_states, dtype=np.float32)
                    observation["eef_states"] = np.array(eef_states, dtype=np.float32)
                    observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)
                    observation["robot_states"] = np.array(actions_segment[:,:-1], dtype=np.float32) # Approximation - might need adjustment if robot_states are different

                    segmented_observations[skill_name] = observation
                    segmented_states[skill_name] = states_segment
                    segmented_actions[skill_name] = actions_segment
                    segmented_rewards[skill_name] = rewards_segment


                # save segmented data
                save_demo_path = save_benchmark_path / (
                    str(task_file).split("/")[-1][:-10] + "_" + demo + "_segmented.pkl"
                )
                with open(save_demo_path, "wb") as f:
                    pkl.dump(
                        {
                            "segmented_observations": segmented_observations,
                            "segmented_states": segmented_states,
                            "segmented_actions": segmented_actions,
                            "segmented_rewards": segmented_rewards,
                            "task_emb": lang_model.encode(task_instruction), # Encode task instruction for the current skill segment
                            "skills": skills,
                            "task_instructions": task_instructions # Save all task instructions for debugging/reference
                        },
                        f,
                    )
                print(f"Saved segmented demo to {str(save_demo_path)}")
            else:
                print(f"No skill segmentation found for task {task_name_from_file}, demo {demo}. Saving full demo.")
                observations = []
                states = []
                actions = []
                rewards = []

                observation = {}
                observation["robot_states"] = np.array(
                    demo_data["robot_states"], dtype=np.float32
                )

                # render image offscreen
                pixels, pixels_ego = [], []
                joint_states, eef_states, gripper_states = [], [], []
                for i in range(len(demo_data["states"])):
                    obs = env.regenerate_obs_from_state(demo_data["states"][i])
                    img = obs["agentview_image"][::-1]
                    img_ego = obs["robot0_eye_in_hand_image"][::-1]
                    joint_state = obs["robot0_joint_pos"]
                    eef_state = np.concatenate(
                        [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                    )
                    gripper_state = obs["robot0_gripper_qpos"]
                    # append
                    pixels.append(img)
                    pixels_ego.append(img_ego)
                    joint_states.append(joint_state)
                    eef_states.append(eef_state)
                    gripper_states.append(gripper_state)
                observation["pixels"] = np.array(pixels, dtype=np.uint8)
                observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
                observation["joint_states"] = np.array(joint_states, dtype=np.float32)
                observation["eef_states"] = np.array(eef_states, dtype=np.float32)
                observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)

                observations.append(observation)
                states.append(np.array(demo_data["states"], dtype=np.float32))
                actions.append(np.array(demo_data["actions"], dtype=np.float32))
                rewards.append(np.array(demo_data["rewards"], dtype=np.float32))


                # save data - full demo if no segmentation found
                save_data_path = save_benchmark_path / (
                    str(task_file).split("/")[-1][:-10] + "_" + demo + "_full.pkl" # Saved as full demo
                )
                with open(save_data_path, "wb") as f:
                    pkl.dump(
                        {
                            "observations": observations,
                            "states": states,
                            "actions": actions,
                            "rewards": rewards,
                            "task_emb": lang_model.encode(env.language_instruction), # Still using the last env's language instruction if no segmentation. Consider using the overall task instruction if needed.
                        },
                        f,
                    )
                print(f"Saved full demo to {str(save_data_path)}")


        tasks_stored += 1
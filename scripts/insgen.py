import os
import json
import random
import datasets
from PIL import Image

class SceneConfig(datasets.BuilderConfig):
    def __init__(self, tasks, modes, data_dir, **kwargs):
        super(SceneConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir

class SceneDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = SceneConfig
    BUILDER_CONFIGS = [
        SceneConfig(
            name="R2R_Goal",
            version="1.0.0",
            description="Scene-level instruction generation and visualization.",
            tasks=["navigation_simulation"],
            modes=["single_step_visualization", "instruction_gen_visualcues", "task_level_evaluation"],
            data_dir="R2R_Goal",
        )
    ]
    DEFAULT_CONFIG_NAME = "R2R_Goal"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'idx': datasets.Value('int32'),
                'scene_id': datasets.Value('string'),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                'train_task': datasets.Value("string"),
            })
        )

    def _split_generators(self, dl_manager):
        data_root = os.path.join(self.config.data_dir, "R2R_Goal")
        print(f"data root: {data_root}")
        all_possible_traj = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

        all_traj_names = []
        for traj_name in all_possible_traj:
            traj_path = os.path.join(data_root, traj_name)
            try:
                # num_images = len([f for f in os.listdir(traj_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                all_traj_names.append(traj_name)
            except OSError as e:
                print(f"Could not access {traj_path}: {e}")

        print(f"Found {len(all_traj_names)} trajectories with length x.")
        # all_traj_names = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        random.seed(42)
        random.shuffle(all_traj_names)

        total_used_ratio = 1  # Use 100% of the data, change to 0.5 to use 50%
        train_ratio, dev_ratio, test_ratio = 0.6, 0.2, 0.2

        # total_used_ratio = 1  # Use 100% of the data, change to 0.5 to use 50%
        # train_ratio, dev_ratio, test_ratio = 0.96, 0.02, 0.02
        total_count = int(len(all_traj_names) * total_used_ratio)
        split_point1 = int(total_count * train_ratio)
        split_point2 = split_point1 + int(total_count * dev_ratio)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "scene_dirs": [os.path.join(data_root, name) for name in all_traj_names[:split_point1]],
                    "split": datasets.Split.TRAIN 
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={
                    "scene_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point1:split_point2]],
                    "split": datasets.Split.VALIDATION
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={
                    "scene_dirs": [os.path.join(data_root, name) for name in all_traj_names[split_point2:total_count]],
                    "split": datasets.Split.TEST
                }
            ),
        ]

    def _load_scene_data(self, scene_dir):
        image_files = sorted([f for f in os.listdir(scene_dir) if f.endswith(".jpg") and f.startswith("frame_")])
        images = [Image.open(os.path.join(scene_dir, f)).convert("RGB").resize((256, 256)) for f in image_files]
        image_paths = [os.path.join(scene_dir, f) for f in image_files]
        with open(os.path.join(scene_dir, "ins.json"), "r") as f:
            instruction = json.load(f)["instruction"]
        return images, image_paths, instruction

    def _prepare_instruction_sample(self, scene_id, images, image_paths, instruction):
        total = len(images)
        if total < 2:
            return None  # too short to process

        start_idx = 0
        end_idx = total - 1

        # Always include start and end frames
        selected_indices = {start_idx, end_idx}

        # Sample additional frames from the middle
        num_to_sample = min(4, total)
        remaining_indices = list(set(range(1, end_idx)) - selected_indices)
        num_additional = num_to_sample - len(selected_indices)

        if remaining_indices and num_additional > 0:
            sampled_middle = random.sample(remaining_indices, min(num_additional, len(remaining_indices)))
            selected_indices.update(sampled_middle)

        indices = sorted(selected_indices)
        sampled_imgs = [images[i] for i in indices]
        sampled_paths = [image_paths[i] for i in indices]

        num_intermediate = len(sampled_imgs) - 2  
        intermediate_str = " ".join(["<image>"] * num_intermediate)

        input_text = (
            "Task: Scene-level Instruction Generation\n"
            "Description: Given a sequence of sampled first-person observations along a navigation trajectory — including the starting point, several intermediate steps, and the goal point — generate a natural language instruction describing how to navigate from the start to the goal.\n"
            "Input Observations:\n"
            "- Start Point: <image>\n"
            f"- Intermediate Points: {intermediate_str}\n"
            "- Goal Point: <image>"
        )

        return {
            "scene_id": scene_id,
            "input_text": input_text,
            "input_imgs": sampled_imgs,
            "input_img_paths": sampled_paths,
            "label_text": instruction,
            "label_imgs": [],
            "label_img_paths": [],
            "train_task": "instruction_gen_visualcues",
        }

    def _prepare_visualization_sample(self, scene_id, images, image_paths, start_idx):
        goal_img = images[-1]
        goal_path = image_paths[-1]
        input_imgs = [images[start_idx], images[start_idx + 1], goal_img]
        input_paths = [image_paths[start_idx], image_paths[start_idx + 1], goal_path]
        label_img = images[start_idx + 2]
        label_path = image_paths[start_idx + 2]

        num_current = len(input_imgs) - 2  
        current_str = " ".join(["<image>"] * num_current)

        input_text = (
            "Task: Navigation Single Step Visualization\n"
            "Description: Given one first-person observations from the recent past, the current first-person observation, and the goal observation, predict the next first-person observation the agent would see if it continues toward the goal.\n"
            "Input observations:\n"
            f"- Previous observation: <image>\n"
            "- Current observation: <image>\n"
            "- Goal observation: <image>"
        )

        return {
            "scene_id": scene_id,
            "input_text": input_text,
            "input_imgs": input_imgs,
            "input_img_paths": input_paths,
            "label_text": "<image>",
            "label_imgs": [label_img],
            "label_img_paths": [label_path],
            "train_task": "single_step_visualization",
        }

    def _prepare_task_level_sample(self, scene_id, images, image_paths):
        # 简单复用 visualization 的最后一帧
        goal_img = images[-1]
        goal_path = image_paths[-1]
        input_imgs = images[:6] + [goal_img]
        input_paths = image_paths[:6] + [goal_img]

        input_text = (
            "Input Sequence: " + " ".join(["<image>"] * len(input_imgs))
        )

        return {
            "scene_id": scene_id,
            "input_text": input_text,
            "input_imgs": input_imgs,
            "input_img_paths": input_paths,
            "label_text": "<image>",
            "label_imgs": [goal_img],
            "label_img_paths": [goal_path],
            "train_task": "task_level_evaluation",
        }

    def _generate_examples(self, scene_dirs, split):
        global_idx = 0

        print(f"[Generate] Split: {split}, Num scenes: {len(scene_dirs)}")
        for scene_dir_name in scene_dirs:
            scene_dir = scene_dir_name
            scene_id = scene_dir_name

            images, image_paths, instruction = self._load_scene_data(scene_dir)
            total_frames = len(images)

            # Instruction Generation + Visualization Interleaving
            if split != datasets.Split.TEST and ("instruction_gen_visualcues" in self.config.modes or "single_step_visualization" in self.config.modes):
                num_action_samples = 5
                num_visual_samples = min(8, max(0, total_frames - 3))
                max_samples = max(num_action_samples, num_visual_samples)

                for i in range(max_samples):
                    # Instruction Generation
                    if i < num_action_samples and "instruction_gen_visualcues" in self.config.modes:
                        sample = self._prepare_instruction_sample(scene_id, images, image_paths, instruction)
                        sample['idx'] = global_idx
                        yield global_idx, sample
                        global_idx += 1

                    # Visualization
                    if i < num_visual_samples and "single_step_visualization" in self.config.modes:
                        sample = self._prepare_visualization_sample(scene_id, images, image_paths, i)
                        sample['idx'] = global_idx
                        yield global_idx, sample
                        global_idx += 1

            # Task-Level
            if split == datasets.Split.TEST and "task_level_evaluation" in self.config.modes:
                sample = self._prepare_task_level_sample(scene_id, images, image_paths)
                sample['idx'] = global_idx
                yield global_idx, sample
                global_idx += 1

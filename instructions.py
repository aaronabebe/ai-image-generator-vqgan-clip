import argparse
import glob
import csv
import os
import uuid


class InstructionPrompt:
    def __init__(self, folder_id, init_image, prompt: str, first_frame, last_frame):
        self.init_image = init_image
        self.prompt = prompt
        self.first_frame = int(first_frame)
        self.last_frame = int(last_frame)
        self.current_frame = self.first_frame
        self.uuid = folder_id

    def iter(self):
        for i in range(self.first_frame, self.last_frame + 1):
            self.current_frame = i
            yield i, self

    def args_override(self):
        return argparse.Namespace(
            prompts=[self.prompt],
            init_image=self.init_image,
        )

    def save_path(self):
        return f'results/{self.uuid}/{self.current_frame}_{self.prompt.replace(" ", "_")[:50]}.png'  # 30 chars max to prevent path length issues

    def last_path(self):
        return f'results/{self.uuid}/{self.last_frame}_{self.prompt.replace(" ", "_")}.png'

    def __str__(self):
        return f"InstructionPrompt(init_image={self.init_image}, prompt={self.prompt}, first_frame={self.first_frame}, last_frame={self.last_frame})"

    def __repr__(self):
        return f"InstructionPrompt(init_image={self.init_image}, prompt={self.prompt}, first_frame={self.first_frame}, last_frame={self.last_frame})"


class Instructions:
    def __init__(self, uuid):
        self.prompts = []
        self.uuid = uuid
        os.makedirs(f'results/{self.uuid}', exist_ok=True)

    @classmethod
    def from_csv(cls, path, continue_from_id=None):
        if continue_from_id:
            instructions = cls(continue_from_id)

            last_file = max(map(os.path.basename, glob.glob(f'results/{continue_from_id}/*.png')),
                            key=lambda x: int(x.split('_')[0]))
            last_iteration = int(last_file.split('_')[0])

            run_once = True
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if int(row[0]) > last_iteration:
                        first_frame = int(row[0])
                        last_frame = int(row[0])

                        if run_once:
                            run_once_init_image = row[2] if len(
                                row) > 2 else f'results/{continue_from_id}/{str(last_prompt_frame)}_{"_".join(last_file.split("_")[1:])}'
                            instructions.prompts.append(
                                InstructionPrompt(
                                    folder_id=continue_from_id,
                                    init_image=run_once_init_image,
                                    prompt=last_prompt,
                                    first_frame=last_prompt_frame,
                                    last_frame=first_frame - 1
                                )
                            )
                            run_once = False

                        if reached_last_cell(row):
                            instructions.prompts[-1].last_frame = first_frame
                            break

                        if instructions.prompts:
                            instructions.prompts[-1].last_frame = first_frame - 1

                        init_image = get_init_image_path(
                            instructions=instructions,
                            row=row,
                            path_override=f'results/{str(last_prompt_frame)}_{"_".join(last_file.split("_")[1:])}'
                        )

                        prompt = row[1]
                        instructions.prompts.append(
                            InstructionPrompt(continue_from_id, init_image, prompt, first_frame, last_frame))
                    else:
                        if len(row) < 2:
                            continue
                        last_prompt_frame = row[0]
                        last_prompt = row[1]
            return instructions

        id = uuid.uuid4()
        instructions = cls(id)

        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                first_frame = int(row[0])
                last_frame = int(row[0])

                if reached_last_cell(row):
                    instructions.prompts[-1].last_frame = first_frame
                    break

                if instructions.prompts:
                    instructions.prompts[-1].last_frame = first_frame - 1

                init_image = get_init_image_path(instructions, row)
                prompt = row[1]
                instructions.prompts.append(InstructionPrompt(id, init_image, prompt, first_frame, last_frame))
        return instructions


def reached_last_cell(row) -> bool:
    return len(row) < 2


def get_init_image_path(instructions, row, path_override=None) -> str:
    if len(row) == 3 and os.path.exists(row[2]):
        return row[2]

    if instructions.prompts:
        return instructions.prompts[-1].last_path()

    return path_override

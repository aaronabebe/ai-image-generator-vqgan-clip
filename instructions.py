import argparse
import csv
import os.path
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
        return f'results/{self.uuid}/{self.current_frame}_progress_{self.prompt.replace(" ", "_")}.png'

    def last_path(self):
        return f'results/{self.uuid}/{self.last_frame}_progress_{self.prompt.replace(" ", "_")}.png'

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
    def from_csv(cls, path):
        id = uuid.uuid4()
        instructions = cls(id)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                first_frame = row[0]
                last_frame = row[0]

                if len(row) < 2:
                    # reached last cell
                    instructions.prompts[-1].last_frame = int(first_frame)
                    break

                if instructions.prompts:
                    instructions.prompts[-1].last_frame = int(first_frame) - 1

                if len(row) == 3 and os.path.exists(row[2]):
                    init_image = row[2]
                else:
                    if instructions.prompts:
                        init_image = instructions.prompts[-1].last_path()
                    else:
                        init_image = None

                prompt = row[1]
                instructions.prompts.append(InstructionPrompt(id, init_image, prompt, first_frame, last_frame))
            print(instructions.prompts)
        return instructions

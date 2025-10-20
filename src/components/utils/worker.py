from src.components.enviroments.thor_env import ThorEnv


class Worker:
    def __init__(self, connection, index, map_version="metric_semantic_v1"):
        """
        Initializes the worker and runs it.

                Parameters:
                        conf (dict): Configuration-Dictionary of the environment.
                        connection (_ConnectionBase): Child-Connection of a Pipe
                        index (int): Unique index of the worker.
        """

        self.worker_index = index
        self.connection = connection
        self.env = None
        self.map_version = map_version
        if self.start_env():
            self.run()

    def start_env(self):
        # Helper to start or restart the ThorEnv
        try:
            if self.env is not None:
                self.env.close()
            self.env = ThorEnv(render=False, map_version=self.map_version)
            return True
        except Exception as e:
            print(f"[Worker {self.worker_index}] CRITICAL: Failed to start environment: {e}")
            return False

    def run(self):
        """
        Main loop for handling commands from the main process.

        Waits for messages such as "reset", "step", or "close" from the runner via the pipe connection.
        Depending on the command, it either resets the environment, performs an environment step,
        or closes the worker. After processing, it sends the results and its worker index back to the main process.

        The loop runs until a "close" command is received.
        """

        while True:
            try:
                command, value = self.connection.recv()

                if command == "reset":
                    try:
                        observation = self.env.reset(scene_number=value.get("scene_number"))
                        self.connection.send([observation, self.worker_index])
                    except TimeoutError:
                        print(f"[Worker {self.worker_index}] WARNING: Timeout during reset. Attempting to restart env...")
                        if not self.start_env():
                            break  # Exit if restart fails
                        observation = self.env.reset(scene_number=value.get("scene_number"))
                        self.connection.send([observation, self.worker_index])

                elif command == "step":
                    try:
                        observation = self.env.step(value)
                        self.connection.send([observation, self.worker_index])
                    except TimeoutError:
                        print(f"[Worker {self.worker_index}] WARNING: Timeout during step. Attempting to restart env...")
                        if not self.start_env():
                            break
                        self.connection.send([None, self.worker_index])

                elif command == "close":
                    break

            except (EOFError, BrokenPipeError):
                break

        if self.env:
            self.env.close()
        self.connection.close()
        print(f"Worker {self.worker_index} finished.")

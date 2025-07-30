import threading
from time import sleep
import queues

class ResultHolder:
    def __init__(self):
        self.result = None
        self.ready = threading.Event()

    def set_result(self, value):
        self.result = value
        self.ready.set()

    def get(self, timeout=None):
        self.ready.wait(timeout)
        return self.result

class task_queue:
    def __init__(self):
        self.memory = queues.queue()
        self.lock = threading.Lock()
        self.can_work = threading.Event()
        self.stop_requested = False  #Shutdown flag
        self.work_thread = threading.Thread(target=self.working_thread)
        self.work_thread.start()

    def add_task(self, func, arguments=None):
        result_holder = ResultHolder()
        with self.lock:
            if arguments:
                self.memory.push((1, func, arguments, result_holder))
            else:
                self.memory.push((0, func, None, result_holder))
        return result_holder

    def pop_task(self):
        with self.lock:
            if not self.memory.is_empty():
                return self.memory.pop()
            return None

    def working_thread(self):
        while not self.stop_requested:
            self.can_work.wait()
            while self.can_work.is_set() and not self.stop_requested:
                task = self.pop_task()
                if task:
                    try:
                        if task[0] == 1:
                            result = task[1](*task[2])
                        else:
                            result = task[1]()
                        task[3].set_result(result)
                    except Exception as e:
                        task[3].set_result(e)
                else:
                    sleep(0.1)

    def start_work(self):
        self.can_work.set()

    def stop_work(self):
        self.can_work.clear()

    def shutdown(self):
        self.stop_requested = True
        self.can_work.set()  # Unblocks `wait()` so the thread can exit
        self.work_thread.join()
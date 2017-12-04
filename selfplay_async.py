"""Classes coordinating asynchronous self-play and state evaluation."""
import threading
import multiprocessing
import os
import Queue

import numpy as np

import gpu_batching_mp
import rngseeder
import selfplay_taskgen
import utils

### PLAYER PROCESSES ###
# The player processes acquire tasks and post results in the respective
# queues. They also have a pipe, through which they can request board
# evaluations, and a pipe through which they can receive signals.
# These signals can be commands to:
# - stop the current task and pause.
# - resume and get a new task
# - stop the current task and terminate the process.

Q_GET_TIMEOUT = 0.05


_SIGNAL_STOP = '<signal_stop>'
_SIGNAL_PAUSE_CLEAN = '<signal_pause_clean>'
_SIGNAL_PROC_PAUSED = '<signal_proc_paused>'
_SIGNAL_RESUME = '<signal_resume>'


def empty_queue(queue):
  items = []
  try:
    while True:
      items.append(queue.get(block=False))
  except Queue.Empty:
    return items


class PlayerProcess(multiprocessing.Process):

  def __init__(self, tasks_queue, results_queue, eval_pipe, process_id=None):
    name = 'player_%d' % process_id if process_id else 'player'
    super(PlayerProcess, self).__init__(name=name)
    self.daemon = True
    self._process_id = process_id
    self._tasks_queue = tasks_queue
    self._results_queue = results_queue
    self._eval_pipe =  eval_pipe
    # Initialise synchronisation primitives
    # We use a pipe to send messages, instead of a combination of events and
    # conditions, for simplicity.
    self._conn_main, self._conn_player = multiprocessing.Pipe()
    # Attributes for the interface to the main process
    self._is_signaled_to_stop = False
    self._is_signaled_to_pause = False
    self._is_paused = False

  def _poll_for_signal(self):
    """Poll for a signal sent from the main process."""
    if not self._conn_player.poll():
      return None
    else:
      return self._conn_player.recv()

  def _post_result(self, result):
    self._results_queue.put(result)

  def _signal_paused(self):
    self._conn_player.send(_SIGNAL_PROC_PAUSED)

  def _wait_signal_resume(self):
    signal = self._conn_player.recv()
    assert signal == _SIGNAL_RESUME

  def _try_get_task(self):
    """Try to acquire a task from the tasks queue"""
    try:
      task = self._tasks_queue.get(block=True, timeout=Q_GET_TIMEOUT)
      return task
    except Queue.Empty:
      return None

  def run(self):

    task = None
    flag_stop = False
    while not flag_stop:
      # Poll for a signal
      signal = self._poll_for_signal()
      # Interpret the signal
      if signal:
        assert signal in (_SIGNAL_STOP, _SIGNAL_PAUSE_CLEAN)
        # Case signal stop:
        if signal == _SIGNAL_STOP:
          # Stop the current task prematurely, and end the process
          if task is not None:
            self._post_result(task.get_result())
          task = None
          flag_stop = True
          continue
        # Case signal pause clean
        elif signal == _SIGNAL_PAUSE_CLEAN:
          # Stop the current task prematurely and wait for the signal
          # to continue. Also signal the main process that you were paused.
          if task is not None:
            self._post_result(task.get_result())
          task = None
          self._signal_paused()
          self._wait_signal_resume()
          continue

      # If we don't have a task, try get a new one
      if not task:
        task = self._try_get_task()
        if task:
          task.init_task(self._eval_pipe)

      # Do the task (if we have one)
      if task:
        task.do_task_step()

      # Post the result and remove the task
      if task and task.is_finished():
        self._post_result(task.get_result())
        task = None


  ## Interface for the main process, to send signals
  def signal_stop(self):
    if not self._is_signaled_to_stop:
      self._conn_main.send(_SIGNAL_STOP)
      self._is_signaled_to_stop = True

  def stop(self):
    self.signal_stop()
    self.join()

  def signal_pause(self):
    if not self._is_signaled_to_pause:
      self._conn_main.send(_SIGNAL_PAUSE_CLEAN)
      self._is_signaled_to_pause = True

  def _receive_pause_ack(self):
    if self._is_signaled_to_pause and not self._is_paused:
      assert self._conn_main.recv() == _SIGNAL_PROC_PAUSED  # blocking
      self._is_paused = True

  def pause(self):
    self.signal_pause()
    self._receive_pause_ack()

  def signal_resume(self):
    assert self._is_paused and self._is_signaled_to_pause
    self._conn_main.send(_SIGNAL_RESUME)
    self._is_paused = self._is_signaled_to_pause = False


### MAIN PROCESS ###

# Controller thread
# The controller thread is responsible for gathering the results from the player
# processes, and creating new tasks.
# The controller thread can be paused or stopped.
#
# A thread is used instead of a process, since it's operations are not critical
# (results only come in relatively sparsely), and it makes it easier to adjust
# the parameters of the task creator.

_CONTROLLER_TIMEOUT = 0.2

class ControllerThread(threading.Thread):

  def __init__(self, results_queue, tasks_queue,
               results_buffer, task_creator, resume_after_break):
    super(ControllerThread, self).__init__()
    self.task_creator = task_creator
    # Interface with players (producer - consumer queues)
    self._results_queue = results_queue  # consumer results
    self._tasks_queue = tasks_queue  # producer tasks
    # initialise synchronisation primitives with main thread
    self._stop_event = threading.Event()
    self._pause_lock = threading.Lock()
    self._results_buffer = results_buffer

    self._resume_after_break = resume_after_break

  def _try_get_result(self):
    """Try to acquire a result from the results queue."""
    try:
      result = self._results_queue.get(
          block=True, timeout=Q_GET_TIMEOUT)
      return result
    except Queue.Empty:
      return None

  def _empty_queue(self):
    empty_queue(self._results_queue)

  def _should_stop(self):
    return self._stop_event.is_set()

  def _dump_result(self, result):
    self._results_buffer.put(result)

  def run(self):
    # Increment this process' niceness
    os.nice(2)
    # We continue polling for results untill the stop_evant flag is set.
    while not self._should_stop():
      # If the pause lock is acquired, it means the main thread is
      # changing the networks and or tasks, and you should pause.
      with self._pause_lock:
        # First, try get a result
        result = self._try_get_result()
        if result is None:
          continue
        self._dump_result(result)
        # If a result was acquired, push a new task
        if result['result'] in ('*', None) and result['positions']:
          task = self.task_creator.create_task(
              resuming_state=result['positions'][-1])
        else:
          task = self.task_creator.create_task()
        self._tasks_queue.put(task)
    # Empty the queue before stopping
    self._empty_queue()

  def signal_stop(self):
    self._stop_event.set()

  def pause(self):
    self._pause_lock.acquire()

  def unpause(self):
    self._pause_lock.release()



# Public interface

NUM_TASKS_PER_PLAYER = 2

class AsyncSelfPlayers(object):

  def __init__(self, eval_networks, eval_batch_size, nr_processes,
               task_creator_kwargs,
               resume_after_break=True):

    self._eval_networks = eval_networks
    self._eval_batch_size = eval_batch_size
    self._nr_processes = nr_processes
    self._resume_after_break = resume_after_break

    self._setup_network_spoolers()
    self._setup_players_and_controller(task_creator_kwargs)

  def _setup_network_spoolers(self):
    """Initialise and start the gpu spoolers.

    The network spoolers act as intermediates for the players to
    use the evaluation networks. Each network gets its own spooler.
    """
    self._gpu_buffer = gpu_batching_mp.GPUBufferMP(
        eval_funcs=self._eval_networks,
        eval_batch_size=self._eval_batch_size)
    self._connections = self._gpu_buffer.create_new_pipes(self._nr_processes)

  def _setup_players_and_controller(self, task_creator_kwargs):
    # Create producer and consumer queues for the controller thread
    self._results_queue = multiprocessing.Queue()
    self._tasks_queue = multiprocessing.Queue()
    # Create buffer and task createor
    self._buffer = utils.ThreadSafeBuffer()
    self._task_creator = selfplay_taskgen.TaskCreator(**task_creator_kwargs)
    # spawn the processes and controller thread
    self._controller = ControllerThread(
        self._results_queue, self._tasks_queue, self._buffer,
        self._task_creator, self._resume_after_break
        )
    self._player_processes = [
        PlayerProcess(self._tasks_queue, self._results_queue, conn, process_id=idx)
        for idx, conn in enumerate(self._connections)]
    # Populate the tasks buffer before starting the processes
    for _ in xrange(NUM_TASKS_PER_PLAYER * len(self._player_processes)):
      self._tasks_queue.put(self._task_creator.create_task())
    # Start the processes and threads
    self._controller.start()
    for player in self._player_processes:
      player.start()

  def _pause_everything(self):
    # Give the pause signal in advance
    for player in self._player_processes:
      player.signal_pause()
    # Wait for the controller to pause
    self._controller.pause()
    # Wait for the players to pause
    for player in self._player_processes:
      player.pause()

  def _resume_everything(self):
    for player in self._player_processes:
      player.signal_resume()
    self._controller.unpause()

  def get_games(self):
    return self._buffer.get_all()

  def update_networks_and_players(self, new_parameters,
                                  new_players, new_matchups):
    # Pause players and controller
    self._pause_everything()
    # update everything
    self._task_creator.change_matchups(
        players=new_players, matchups=new_matchups)
    for network, params in zip(self._eval_networks, new_parameters):
      if params is not None:
        network.set_params(params)
    # Refill the task buffer
    tasks = empty_queue(self._tasks_queue)
    for task in tasks:
      self._tasks_queue.put(self._task_creator.create_task(resuming_task=task))
    # Now make the changes
    self._resume_everything()


  def stop(self):
    # Give the stop signal in advance
    for player in self._player_processes:
      player.signal_stop()
    # Wait for the players to stop
    # They are stopped before the controller thread,
    # since the controller needs to empty their results queue
    for i, player in enumerate(self._player_processes):
      player.join()
    # Now stop the controller
    self._controller.signal_stop()
    empty_queue(self._tasks_queue)
    self._controller.join()
    # signal the buffer to stop last
    self._gpu_buffer.stop()



### EVALUATION OF SINGLE POSITIONS ###


class EvaluationProcess(multiprocessing.Process):

  def __init__(self, tasks_queue, results_queue, eval_pipe,
               targetcomputer, seed, process_id=None):
    name = 'targetcomputer_%d' % process_id if process_id else 'targetcomputer'
    super(EvaluationProcess, self).__init__(name=name)
    self.daemon = True
    self._process_id = process_id
    self._tasks_queue = tasks_queue
    self._results_queue = results_queue
    # Communication primitives
    self._flag_stop = multiprocessing.Event()
    self._flag_is_stopping = multiprocessing.Event()
    self._is_signaled_to_stop = False

    self._targetcomputer = targetcomputer
    self._rng = np.random.RandomState(seed)
    self._eval_pipe =  eval_pipe

  def _post_result(self, result):
    self._results_queue.put(result)

  def _try_get_task(self):
    """Try to acquire a task from the tasks queue"""
    try:
      task = self._tasks_queue.get(block=True, timeout=Q_GET_TIMEOUT)
      return task
    except Queue.Empty:
      return None

  def run(self):
    # the main thread relies on these processes to return results for the batch
    # quickly, so we give them slightly higher priority
    os.nice(1)
    task = None
    self._targetcomputer.init(self._eval_pipe, self._rng)
    while not self._flag_stop.is_set():
      task = self._try_get_task()
      if task:
        result = self._targetcomputer(task)
        self._post_result(result)
        task = None
    self._flag_is_stopping.set()

  ## Interface for the main process, to send signals
  def signal_stop(self):
    if not self._is_signaled_to_stop:
      self._flag_stop.set()
      self._is_signaled_to_stop = True

  def wait_stopping(self):
    self.signal_stop()
    self._flag_is_stopping.wait()

  def stop(self):
    self.signal_stop()
    self.join()


class AsyncTargetComputers(object):

  def __init__(self, eval_network, eval_batch_size, nr_processes, targetcomputer,
               seed):
    self._eval_network = eval_network
    self._eval_batch_size = eval_batch_size
    self._nr_processes = nr_processes
    self._targetcomputer = targetcomputer
    self._seeder = rngseeder.RNGSeeder(seed)

    self._setup_network_spooler()
    self._setup_players()

  def _setup_network_spooler(self):
    """Initialise and start the gpu spooler."""
    self._gpu_buffer = gpu_batching_mp.GPUBufferMP(
        eval_funcs=[self._eval_network],
        eval_batch_size=self._eval_batch_size)
    self._connections = self._gpu_buffer.create_new_pipes(self._nr_processes)

  def _setup_players(self):
    # Create producer and consumer queues for the controller thread
    self._results_queue = multiprocessing.Queue()
    self._tasks_queue = multiprocessing.Queue()
    # spawn the processess
    self._targetcomputer_processes = [
        EvaluationProcess(
            self._tasks_queue, self._results_queue, conn,
            self._targetcomputer, self._seeder(), process_id=None)
        for idx, conn in enumerate(self._connections)]
    # Start the processes and threads
    for proc in self._targetcomputer_processes:
      proc.start()

  def eval_samples(self, samples):
    # dump into queue
    for sample in samples:
      self._tasks_queue.put(sample)
    # wait for results
    results = []
    for _ in samples:
      res = self._results_queue.get()
      results.append(res)

    return results

  def stop(self):
    for proc in self._targetcomputer_processes:
      proc.signal_stop()
    for proc in self._targetcomputer_processes:
      proc.wait_stopping()
    self._gpu_buffer.stop()
    empty_queue(self._results_queue)
    for proc in self._targetcomputer_processes:
      proc.stop()

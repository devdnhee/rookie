"""A threadsafe queue with an efficient way of pulling batches."""

import collections
import Queue
import threading
import time

import tf_nn_utils


class BatchingQueue(object):

  def __init__(self, batch_size):
    self.batch_size = batch_size
    self._items_queue = []
    self._items_mutex = threading.RLock()
    self._batches_queue = collections.deque()
    self._batches_mutex = threading.RLock()

    self._items_not_empty = threading.Condition(self._items_mutex)

    self._init(batch_size)

  def _move_batch(self):
    self._items_mutex.acquire()
    try:
      # Check if there are items available, otherwise do nothing
      if not self._items_queue:
        return 
      else:
        self._batches_mutex.acquire()
        try:
          # Get the items from the queue
          items = self._items_queue
          self._items_queue = []
          # Batch them
          batch = self._batch_items(items)
          # Add to batch queue
          self._batches_queue.append(batch)
        finally:
          self._batches_mutex.release()
    finally:
      self._items_mutex.release()

  def put(self, item):
    """Put an item into the queue."""
    self._items_not_empty.acquire()
    try:
      # Simply add the item to the queue
      self._items_queue.append(item)
      self._items_not_empty.notify()
      # If the batch is full, add the items to the batch queue.not_empty
      if len(self._items_queue) == self.batch_size:
        self._move_batch()
    finally:
      self._items_not_empty.release()

  def get_batch(self, block=True, timeout=None):
    """Get a batch from the queue."""
    # First, try and get a batch from the batches queue, without locking the
    # items queue. This allows for better concurrency.
    self._batches_mutex.acquire()
    try:
      if self._batches_queue:
        return self._batches_queue.popleft()
    finally:
      self._batches_mutex.release()
    # If it failed (the queue was empty), try to make a batch from items
    # in the other queue. Before doing so, we should check if no batch was added
    # in the meantime.
    self._items_not_empty.acquire()
    self._batches_mutex.acquire()
    try:
      # Acquiring both locks assures us that no thread is currently trying to
      # make a batch. Note that we still need to check the batches queue. A 
      # batch might have been added while we had no hold over the locks.
      if not block:
        if not (self._batches_queue or self._items_queue):
          raise Queue.Empty
      elif timeout is None:
        while not (self._batches_queue or self._items_queue):
          self._batches_mutex.release()
          self._items_not_empty.wait()
          self._batches_mutex.acquire()
      elif timeout < 0:
        raise ValueError("'timeout' must be a non-negative number")
      else:
        endtime = time.time() + timeout
        while not (self._batches_queue or self._items_queue):
          remaining = endtime - time.time()
          if remaining <= 0.0:
            raise Queue.Empty
          self._batches_mutex.release()
          self._items_not_empty.wait(remaining)
          self._batches_mutex.acquire()

      # First we check for batches again.
      if self._batches_queue:
        return self._batches_queue.popleft()
      # If not, we batch the available items ourselves. At this point, there 
      # should be items there.
      assert self._items_queue
      self._move_batch()
      return self._batches_queue.popleft()
    finally:
      self._batches_mutex.release()
      self._items_not_empty.release()

  # Methods to override

  def _init(self, batch_size):
    pass

  def _batch_items(self, items):
    return items


class CountingBatchingQueue(BatchingQueue):

  def _init(self, batch_size):
    self.nr_items = 0
    self.nr_batches = 0

  def _batch_items(self, items):
    self.nr_items += len(items)
    self.nr_batches += 1
    return super(CountingBatchingQueue, self)._batch_items(items)


class ConcatBatchingQueue(BatchingQueue):

  def _batch_items(self, entries):
    datas = [entry.get_sample() for entry in entries]
    batched = tf_nn_utils.combine_into_chunk(datas)
    return entries, batched


class CountingConcatBatchingQueue(ConcatBatchingQueue):

  def _init(self, batch_size):
    self.nr_items = 0
    self.nr_batches = 0

  def _batch_items(self, items):
    self.nr_items += len(items)
    self.nr_batches += 1
    return super(CountingConcatBatchingQueue, self)._batch_items(items)
    

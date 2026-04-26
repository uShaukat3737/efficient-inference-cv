import json
import time
import torch

#dynamically batches requests from the queue
class BatchScheduler:
  def __init__(self,consumer,model,batch_size=8,timeout=0.02):
    self.consumer=consumer
    self.model=model
    self.batch_size=batch_size
    self.timeout=timeout

  def run(self):
    batch=[]
    start_time=time.time()

    while True:
      #enforce timeout condition
      if len(batch)>0 and (time.time()-start_time)>=self.timeout:
        self.process_batch(batch, batch_trigger="timeout")
        batch=[]
        start_time=time.time()

      try:
        #if batch is empty, wait indefinitely for the first request
        if len(batch) == 0:
            req=self.consumer.pop_blocking()
            if req:
                tensor=torch.tensor(req["data"])
                dequeue_time=time.time()
                batch.append({
                    "id": req["id"],
                    "tensor": tensor,
                    "enqueue_time": req.get("enqueue_time"),
                    "dequeue_time": dequeue_time
                })
                start_time=time.time() #start the timeout clock
        #if batch has items, don't block so we can honor the timeout
        else:
            req=self.consumer.pop_non_blocking()
            if req:
                tensor=torch.tensor(req["data"])
                dequeue_time=time.time()
                batch.append({
                    "id": req["id"],
                    "tensor": tensor,
                    "enqueue_time": req.get("enqueue_time"),
                    "dequeue_time": dequeue_time
                })
                #process immediately if batch is full
                if len(batch)>=self.batch_size:
                    self.process_batch(batch, batch_trigger="size")
                    batch=[]
                    start_time=time.time()
            else:
                #sleep 1ms to prevent cpu thrashing
                time.sleep(0.001)

      except Exception as e:
        print(f"scheduler error: {e}")
        time.sleep(0.01)

  def process_batch(self, batch, batch_trigger="size"):
    ids = [item["id"] for item in batch]
    tensors = [item["tensor"] for item in batch]
    enqueue_times = [item["enqueue_time"] for item in batch]
    dequeue_times = [item["dequeue_time"] for item in batch]

    #concat along batch dimension
    batch_tensor=torch.cat(tensors, dim=0)

    try:
        #run batched inference
        inference_start = time.time()
        predictions = self.model.predict_batch(batch_tensor)
        inference_end = time.time()
        total_inference_ms = (inference_end - inference_start) * 1000

        #save results and metrics back to redis
        redis_client = self.consumer.redis
        for i, req_id in enumerate(ids):
            pred = predictions[i]
            #cache result for 60 seconds
            redis_client.setex(f"result:{req_id}", 60, str(pred))

            #compute metrics per request
            enqueue_time = enqueue_times[i]
            dequeue_time = dequeue_times[i]
            queue_wait_ms = (dequeue_time - enqueue_time) * 1000 if enqueue_time else None
            inference_time_ms = total_inference_ms / len(batch)

            metrics = {
                "enqueue_time": enqueue_time,
                "dequeue_time": dequeue_time,
                "queue_wait_ms": queue_wait_ms,
                "inference_time_ms": inference_time_ms,
                "batch_size": len(batch),
                "batch_trigger": batch_trigger,
            }

            #cache metrics for 60 seconds
            redis_client.setex(f"metrics:{req_id}", 60, json.dumps(metrics))
            print(f"[RESULT] {req_id} → {pred}")
    except Exception as e:
        print(f"batch processing failed: {e}")
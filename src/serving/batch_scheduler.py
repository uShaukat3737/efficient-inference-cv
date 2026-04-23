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
        self.process_batch(batch)
        batch=[]
        start_time=time.time()

      try:
        #if batch is empty, wait indefinitely for the first request
        if len(batch) == 0:
            req=self.consumer.pop_blocking()
            if req:
                tensor=torch.tensor(req["data"])
                batch.append((req["id"],tensor))
                start_time=time.time() #start the timeout clock
        #if batch has items, don't block so we can honor the timeout
        else:
            req=self.consumer.pop_non_blocking()
            if req:
                tensor=torch.tensor(req["data"])
                batch.append((req["id"],tensor))
                #process immediately if batch is full
                if len(batch)>=self.batch_size:
                    self.process_batch(batch)
                    batch=[]
                    start_time=time.time()
            else:
                #sleep 1ms to prevent cpu thrashing
                time.sleep(0.001)

      except Exception as e:
        print(f"scheduler error: {e}")
        time.sleep(0.01)

  def process_batch(self,batch):
    ids=[item[0] for item in batch]
    tensors=[item[1] for item in batch]

    #concat along batch dimension
    batch_tensor=torch.cat(tensors, dim=0)

    try:
        #run batched inference
        predictions = self.model.predict_batch(batch_tensor)
        
        #save results back to redis
        redis_client = self.consumer.redis
        for i, req_id in enumerate(ids):
            pred = predictions[i]
            #cache result for 60 seconds
            redis_client.setex(f"result:{req_id}", 60, str(pred))
            print(f"[RESULT] {req_id} → {pred}")
    except Exception as e:
        print(f"batch processing failed: {e}")
class queue:
    #node class
    class node:
        def __init__(self,data,next = None,prev = None):
            self.data = data
            self.next = next
            self.prev = prev
        
        def __str__(self):
            return str(self.data)
            
    def __init__(self,maximum = 100005):
        self.count = 0
        self.head = None
        self.tail = None
        self.maximum = maximum
        
    def is_empty(self):
        if self.count<=0:
            return True
        return False
    
    def is_full(self):
        if self.count>=self.maximum:
            return True
        return False    
    
    def push(self,value):
        if self.is_full():
            raise Exception("full stack")
        
        ele = self.node(value)
        if self.is_empty():
            self.head = ele
            self.tail = ele 
            self.count+=1 
            return None
        
        self.tail.next = ele
        ele.prev = self.tail
        self.tail = ele
        self.count+=1
        return    
    
    def pop(self):
        if self.is_empty():
            raise Exception("EMPTY QUEUE") 
        
        if self.count==1:
            ele = self.head.data
            self.head = self.tail = None
            self.count-=1
            return ele
        
        ele = self.head.data
        next = self.head.next
        next.prev = None
        self.head.next = None
        self.head = next
        self.count-=1
        return ele       
        
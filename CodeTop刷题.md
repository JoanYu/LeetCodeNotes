# CodeTop刷题

网址：https://codetop.cc/



**[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list) （链表的熟悉）**

乍一看，反转链表就是依次替换节点的`next`属性，但是一个节点一替换，它就断了。于是中间需要一个临时变量，先把`next`属性接到临时变量`next`（注意属性`cur.next`和变量`next`的区别）上，再进行替换。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None # 上一个节点
        cur = head # 当前节点
        while cur:
            next = cur.next # next变量
            cur.next = pre # next属性
            pre = cur
            cur = next
        return pre
```



**[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array) & [补充题4. 手撕快速排序](https://leetcode-cn.com/problems/sort-an-array) （快速排序的使用）**

快速排序可以说是算法「八股文」。做了很多基于快排的题，我发现分治部分的代码结构都差不多。

```python
# 分治
def partition(nums, left, right):
        pivot = nums[right]
        i = left - 1 # i从left的前一个开始
        for j in range(left, right):
            if nums[j] < nums[right]:
                i += 1
                nums[j], nums[i] = nums[i], nums[j]
        i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i
```

对于第215题，主函数可以写成：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        size = len(nums)

        target = size - k
        left = 0
        right = size - 1
        while True:
            index = self.__partition(nums, left, right)
            if index == target:
                return nums[index]
            elif index < target:
                # 下一轮在 [index + 1, right] 里找
                left = index + 1
                # 下一轮在 [left, index - 1] 里找
            else:
                right = index - 1

    #  循环不变量：[left + 1, j] < pivot
    #  (j, i) >= pivot
    def __partition(self, nums, left, right):
        pivot = random.randint(left, right)
        nums[pivot], nums[right] = nums[right], nums[pivot]
        i = left - 1
        for j in range(left, right):
            if nums[j] < nums[right]:
                i += 1
                nums[j], nums[i] = nums[i], nums[j]
        i += 1
        nums[i], nums[right] = nums[right], nums[i]
        return i
```

这里在分治算法里加了随机选定`pivot`，使得算法用时快了将近40倍，这是因为测试用例里有极端用例。

对于912题，这种简单的全排序题，最好全文背诵。

```python
class Solution:
    def randomized_partition(self, nums, l, r):
        pivot = random.randint(l, r)
        nums[pivot], nums[r] = nums[r], nums[pivot]
        i = l - 1
        for j in range(l, r):
            if nums[j] < nums[r]:
                i += 1
                nums[j], nums[i] = nums[i], nums[j]
        i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i

    def randomized_quicksort(self, nums, l, r):
        if r - l <= 0:
            return
        mid = self.randomized_partition(nums, l, r)
        self.randomized_quicksort(nums, l, mid - 1)
        self.randomized_quicksort(nums, mid + 1, r)

    def sortArray(self, nums: List[int]) -> List[int]:
        self.randomized_quicksort(nums, 0, len(nums) - 1)
        return nums
```



**[146. LRU缓存机制](https://leetcode-cn.com/problems/lru-cache) (哈希表，双向链表？)**

我自己写的版本没有使用双向链表，用时很长。

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {} # 用字典存储

    def get(self, key: int) -> int:
        if key in self.cache.keys():
          	# 这几步其实是把要使用的键值对移动到最末端
            tmp = self.cache[key]
            self.cache.pop(key)
            self.cache[key] = tmp
            return tmp
        else:
            return -1

    def put(self, key: int, value: int) -> None:
      	# 如果新的键值对是对旧数据的替换，则也需要先移动到最末端
        if key in self.cache.keys():
            tmp = self.cache[key]
            self.cache.pop(key)
        self.cache[key] = value
        # 先放新的，再删旧的
        if len(list(self.cache.keys())) > self.capacity:
            self.cache.pop(list(self.cache.keys())[0])


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```

官方使用了结合哈希表和双向链表的`collections.OrderedDict`，但官方也都说明了不建议使用相关库。

```python
class DLinkedNode:
  	# 这是个带字典的双向链表
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/lru-cache/solution/lruhuan-cun-ji-zhi-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

这个执行速度就很快，很多操作都封装了。



**[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters) （滑动窗口）**

维护两个指针：左指针`i`和右指针`j`，左指针`i`初始化为-1，维护一个哈希表`dic`。

右指针`j`从0开始，如果`s[j]`在字典`dic`里，更新左节点`i`。

然后在哈希表添加{`字母`:`位置`}。

整个算法一直在查找两个相同字符（中间没有不同字符）的距离的最大值。

这样就保证了在最小的时间复杂度里完备地查找了结果。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic, res, i = {}, 0, -1
        for j in range(len(s)):
            if s[j] in dic:
                i = max(dic[s[j]],i)
            dic[s[j]] = j
            res = max(res, j - i)
        return res
```


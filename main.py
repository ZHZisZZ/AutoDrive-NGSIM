def cache(fn):
  cache_info = {}
  def new_fn(n):
    print(cache_info)
    if n in cache_info:
      return cache_info[n]
    result = fn(n)
    cache_info[n] = result
    return result
  return new_fn

@cache
def add(n):
  return n + n

@cache
def multiply(n):
  return n * n


print(add(5))
print(add(5))
print(multiply(5))
print(multiply(5))
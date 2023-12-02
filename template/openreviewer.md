# openreviewer数据格式

### 建议每篇文章的所有数据存到一个字典中，便于处理以及其他后续操作：
```json
{
    "Title": "str",
    "Author": "List[str]",
    "Abstract": "str",
    "Text": "str(main body)",
    "References": "List[str]",
    "Keywords": "List[str]",
    "PublishDate": "datetime",
    "Reviews": "List[str(the first review)](about 3-5 reviews)",
    "Scores": "List[int](same number to Reviews)",
    "Rebuttals": "List[List[str(comment)](a list of comments)](same number to Reviews)",
    "...": "..."
}
```
> 以上只是样例，根据实际更改

### prompt template

以下是一个样例，根据实际更改
```python
prompt = f"You are reviewing the paper titled {Title} by {Author}. The keywords are {Keywords}. The abstract is：\n\n{Abstract}\n\nYou will read this paper and write a review for it. The main body of the paper is: \n\n{Text}\n\nAfter reading this paper, please write your review for it."
```
这种情况下训练时`system prompt`会使用默认的：
```python
system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
```

当然也可以分别够造成 `system prompt` 和 `prompt` ，例如

```python
system_prompt = f"You are reviewing the paper titled {Title} by {Author}. The keywords are {Keywords}. You will read this paper and write a review for it."
```

```python
prompt = f"The abstract is：\n\n{Abstract}\n\nThe main body of the paper is: \n\n{Text}\n\nAfter reading this paper, please write your review for it."
```

### response template

以下是一个样例，根据实际更改
```python
# the i-th response
response = f"{Reviews[i]}\n\nrating:{rating}\n\nconfidence:{confidence}"
```

### 训练数据

```json
{
    "prompt": "str",
    "system_prompt": "Optional[str]",
    "response": "str"
}
```

训练代码需要提供`prompt`，`system_prompt`（可选），`response`。最终会被构造成如下形式：

```python
query = v"{system_prompt} USER: {prompt} ASSISTANT:"
inputs = query + response
```
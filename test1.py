import re




def main(params):
    text = params["text"]
    json_str = ""
    isSuccess = False

    pattern = r'```json(.*?)```'

    mathces = re.findall(pattern, text, re.DOTALL)
    print(f"Matches found: {mathces}")  # 调试输出，查看匹配结果

    if len(mathces) > 0:
        isSuccess = True
        json_str = mathces[0]

    output = {
        "isSuccess": isSuccess,
        "json_str": json_str
    }
    return output

if __name__ == "__main__":
    params = {
        "text": "```json{good: ljr,"
                "bad: zyx}```,"
                "```json{good: ljr,"
                "bad: zyx}```"
    }
    result = main(params)
    print(result)  # 输出结果到控制台
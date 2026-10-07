import ollama
from pydantic import BaseModel, ConfigDict, ValidationError

inventory_db = {
    "laptop": {"stock": 5, "base_price": 1200},
    "monitor": {"stock": 0, "base_price": 300},
    "keyboard": {"stock": 25, "base_price": 80},
}


def check_inventory(product_name):
    product_name = product_name.lower()

    if product_name in inventory_db:
        return inventory_db[product_name]

    return {"stock": 0, "base_price": None}


def calculate_loyalty_discount(base_price, years_as_customer):
    discount = min(years_as_customer * 0.05, 0.30)
    final_price = base_price * (1 - discount)
    return round(final_price, 2)

available_functions = {
    "check_inventory": check_inventory,
    "calculate_loyalty_discount": calculate_loyalty_discount,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Get stock and price for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loyalty_discount",
            "description": "Calculate final price based on loyalty years",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_price": {"type": "number"},
                    "years_as_customer": {"type": "integer"},
                },
                "required": ["base_price", "years_as_customer"],
            },
        },
    },
]

class InventoryArguments(BaseModel):
    model_config = ConfigDict(strict=True)
    product_name: str


class DiscountArguments(BaseModel):
    model_config = ConfigDict(strict=True)
    base_price: float
    years_as_customer: int


argument_models = {
    "check_inventory": InventoryArguments,
    "calculate_loyalty_discount": DiscountArguments,
}


model = "llama3.1:latest"
client = ollama.Client(timeout=120)
message = [
    {"role": "system", "content": "Look up the laptop with check_inventory, then call calculate_loyalty_discount using its base price and the customer years. Do not calculate the discount yourself. Answer after both tool results."},
    {"role": "user", "content": "I am a customer for 5 years. What will be the final price of a laptop?"},
]
executed = []
print("model:", model, flush=True)
print("capabilities:", client.show(model).capabilities, flush=True)

for turn in range(5):
    response = client.chat(
        model=model,
        messages=message,
        tools=tools,
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 256},
    )
    message.append(response.message)
    calls = response.message.tool_calls or []
    print(f"request {turn + 1}: {len(calls)} tool calls", flush=True)
    if not calls:
        print("answer:", response.message.content, flush=True)
        break
    for call in calls:
        name = call.function.name
        args = call.function.arguments
        print("requested:", name, args, flush=True)
        try:
            checked = argument_models[name].model_validate(args).model_dump()
            result = available_functions[name](**checked)
            executed.append((name, result))
        except ValidationError:
            result = {"error": "Use JSON numbers for base_price and years_as_customer, not strings. Retry the function with numeric values."}
        print("returned:", result, flush=True)
        message.append({"role": "tool", "tool_name": name, "content": str(result)})
else:
    print("Stopped: no final answer within five requests.", flush=True)

print("executed tool results:", executed, flush=True)
print("rule for 5 years:", calculate_loyalty_discount(1200, 5), flush=True)
print("rule for 10 years:", calculate_loyalty_discount(1200, 10), flush=True)
client.generate(model=model, keep_alive=0)

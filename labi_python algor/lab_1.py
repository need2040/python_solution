def sums(*args):
    result = 0 
    for num in args:
        result += num
    return result 

def substraction(*args): 
    result = args[0]
    for num in args[1:]:
        result -= num
    return result

def multiplication(*args): 
    result = args[0]
    for num in args[1:]:
        result *= num
    return result

def quotient(*args):
    result = args[0]
    for num in args[1:]:
        if num == 0:  
            return float('inf')
        result /= num
    return result

def cast_to_float(list_obj, round_number):
    for i in range(len(list_obj)):
        list_obj[i] = round(float(list_obj[i]), round_number)

numbers = []
while True:
    user_input = input("Введите число: ")
    if user_input == "":
        break
    numbers.append(user_input)

round_number = int(input("Введите число, до которого необходимо округлить:\n"))
cast_to_float(numbers, round_number)

print(f"Сумма введенных чисел: {sums(*numbers)}")
print("Разница введенных чисел: {}".format(substraction(*numbers)))
print("Произведение ввееденных чисел: %f" % (multiplication(*numbers)))
print(f"Частное введенных чисел: {quotient(*numbers)}")

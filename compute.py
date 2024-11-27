batch = 20
H, W = 192, 192
C = 8



if not batch:
    batch = int(input('batch_size: '))
if not H and not W:
    H, W = list(map(int, input('respolustion: ').split()))
if not C:
    C = int(input('channels: '))
number_states = int(input('sim_num: '))
random_sample = number_states * 2 + 2
all_pairs = (number_states ** 2 + 3 * number_states) // 2


res_1 = 0
res_2 = 0
for layer in range(4):
    res_1 += batch * H * W * C * 17 * random_sample
    res_2 += batch * H * W * C * 17 * all_pairs
    C *= 2
    H //= 2
    W //= 2

print('using_SDS:', res_1)
print('real:', res_2)
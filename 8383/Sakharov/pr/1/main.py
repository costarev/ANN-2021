def calculate_ip(addr, mask):
    address_arr = list(map(int, address.split('.')))
    mask_arr = list(map(int, mask.split('.')))
    neg_mask_arr = [abs(i-255) for i in mask_arr]

    res_network = (".").join(str(i & j) for i, j in zip(mask_arr, address_arr))
    res_broadcast = (".").join(str(i | j) for i, j in zip(neg_mask_arr, address_arr))
    
    return res_network, res_broadcast


print("Ip:")
while True:
    address = str(input())
    if (len(address.split('.')) != 4):
        print("Invalid ip")
    else:
        break

print("Mask:")
while True:
    net_mask = str(input())
    if (len(net_mask.split('.')) != 4):
        print("Invalid mask")
    else:
        break

network, broadcast = calculate_ip(address, net_mask)

print("Network address: " + network)
print("Broadcast address: " + broadcast)

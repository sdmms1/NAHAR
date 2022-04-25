import os

if __name__ == '__main__':
    # with open("./temp.txt", "r") as file:
    #     for e in file.readlines():
    #         if "Model from" in e:
    #             name = e.split(" ")[-1]
    #             name = name[:name.index("-")]
    #             print(name)
    #         elif "Acc" in e:
    #             print(e.split(" ")[-2][:-1])

    with open("./radar_de_dp.txt", "r") as file:
        for e in file.readlines():
            if "Model from" in e:
                name = e.split(" ")[-1]
                name = name[:name.index("-")]
                print(name)
            elif "using different" in e:
                print(e.split(" ")[-2][:-1])
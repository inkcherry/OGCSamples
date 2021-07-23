import  torch
def is_m_w_difference(manager,worker): #check  whether there is overlap between w_para and m_para
    m_paras=manager.parameters()
    w_paras=worker.parameters()
    for mpara in m_paras:
        # print(para)
        torch.zero_(mpara.data)


    for wpara in w_paras:
        wpara.data=torch.ones_like(wpara.data)

    m_paras = manager.parameters()
    w_paras = worker.parameters()
    for mpara in m_paras:
        pp=mpara.data
        # print((pp==1).nonzero())
        print(torch.all(torch.eq(pp, torch.zeros_like(pp))))
    print("-------------------------")


    for mpara in m_paras:
        # print(para)
        torch.zero_(mpara.data)
    for wpara in w_paras:
        pp = wpara.data
        # print((pp == 1).nonzero())
        print(torch.all(torch.eq(pp, torch.ones_like(pp))))



    # if(torch.is_nonzero())



    return


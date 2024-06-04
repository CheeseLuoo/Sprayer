class Network(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 20)
        self.fc5 = torch.nn.Linear(20, 1)

    def forward(self, input_layer):
        hid = torch.tanh(self.fc1(input_layer))
        hid = torch.tanh(self.fc2(hid))
        hid = torch.tanh(self.fc3(hid))
        hid = torch.tanh(self.fc4(hid))
        return self.fc5(hid)


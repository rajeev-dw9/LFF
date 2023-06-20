# import torch
# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, num_classes=10):
#         super(MLP, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(3 * 28*28, 100),
#             nn.ReLU(),
#             nn.Linear(100, 100),
#             nn.ReLU(),
#             nn.Linear(100, 100),
#             nn.ReLU()
#         )
#         self.classifier = nn.Linear(100, num_classes)

    # def forward(self, x, return_feat=True):
    #     x = x.view(x.size(0), -1) / 255
    #     intermediate_feats = []
    #     feat = x = self.feature[0](x)
    #     intermediate_feats.append(feat)

    #     for i in range(1, len(self.feature)):
    #         feat = x = self.feature[i](x)
    #         intermediate_feats.append(feat)

    #     x = self.classifier(x)

    #     if return_feat:
    #         return x, intermediate_feats
    #     else:
    #         return x

    # def forward(self, x, return_feat=True):
    #     x = x.view(x.size(0), -1) / 255
    #     intermediate_feats = []
    #     feat = self.feature[0](x)
    #     intermediate_feats.append(feat)

    #     for i in range(1, len(self.feature)):
    #         feat = self.feature[i](feat)
    #         intermediate_feats.append(feat)

    #     x = self.classifier(feat)

    #     if return_feat:
    #         return x, intermediate_feats
    #     else:
    #         return x



# import torch
# import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, num_classes = 10):
#         super(MLP, self).__init__()  
#         self.l1 = nn.Linear(3 * 28*28, 100)     
#         self.l2 = nn.Linear(100, 100)     
#         self.l3 = nn.Linear(100, 100),    
#         self.classifier = nn.Linear(100, num_classes)

#     def forward(self, x, return_feat=True):
#         x = x.view(x.size(0), -1)# / 255
#         out1 =  self.l1(x).relu() 
#         out2 = self.l2(out1).relu()
#         out3 = self.l3(out2).relu()
#         final = self.classifier(out3)

#         if return_feat:                                                                                                                                                                                            
#             return final, [out1, out2, out3]                                                                                                                                                                        
#         else:                                                                                                                                                                                                      
#             return final
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(3 * 28*28, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, num_classes)

    # def forward(self, x, return_feat=False):
    def forward(self, x, return_feat=True):
        x = x.view(x.size(0), -1) / 255
        out1 = self.l1(x).relu()
        out2 = self.l2(out1).relu()
        out3 = self.l3(out2).relu()
        final = self.classifier(out3)

        if return_feat:
            return final, [out1, out2, out3]
        else:
            return final

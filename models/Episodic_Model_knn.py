"""
Module for episodic (meta) training/testing
"""
import numpy as np
import pandas as pd
from architectures import get_backbone, get_classifier,get_classifier2
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
from utils import getprotoconfi,getknnconfi,protoPred,knn_st,statistic,statistic2,addImages,addImages2,class_balance,calculate_accuracy,isThan16,stastic_balance,statistic_balance
import torch
class EpisodicTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.classifier2 = get_classifier2(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
        self.train_way = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_WAYS
        self.query = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_QUERY

        if config.IS_TRAIN == 0:
            self.support = config.DATA.TEST.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        else:

            self.support = config.DATA.TRAIN.EPISODE_DESCR_CONFIG.NUM_SUPPORT
        self.k = config.K
        self.lam = config.lam
    def forward(self,img_tasks,label_tasks, *args, model, optimizer,step,**kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        loss2 = 0.
        knn_acc = []
        proto_acc = []
        acc = []
        label_acc1 = []
        label_num1 = 0
        label_acc2 = []
        label_num2 = 0
        label_acc = []
        label_num = 0
        label_total_num = 0
        than16_1 = 0
        than16_2 = 0
        than16 = 0
        balance_acc1 = 0
        balance_acc2 = 0
        balance_acc3 = 0
        array = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            if self.support == 1:
                score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                        label_tasks[i]["support"].squeeze_().cuda(), self.k-1, **kwargs)
            else:
                score2, indices2, knn_distances2, knn_pred2, scores2 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features, support_features,label_tasks[i]["support"].squeeze_().cuda(),self.k, **kwargs)

            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = label_tasks[i]["support"].squeeze_().cuda()
            # 如果原型和knn预测一样，并且k近邻中大于三个
            addImage, addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)


            # addImageLabel = label_tasks[i]["query"].squeeze_()[addImage]
            #
            # label_acc1.append(calculate_accuracy(addLabel,addImageLabel))
            # label_num1+=len(addLabel)
            # than16_1 += isThan16(addLabel)

            labelnum = self.train_way * self.query
            temp = self.support*self.train_way
            addLabel2 = addImages2(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features,labelnum,temp)

            addLabel2Index = []
            for addLabel2Index1,addLabel2Index2 in enumerate(addLabel2):
                if(addLabel2Index2>-1):
                    addLabel2Index.append(addLabel2Index1)

            # addLabel2RealLabel = label_tasks[i]["query"].squeeze_()[addLabel2Index]
            #
            # results1 = [addLabel2[q1] for q1 in addLabel2Index]
            # label_acc2.append(calculate_accuracy(results1, addLabel2RealLabel))
            # label_num2+=len(addLabel2Index)
            # than16_2+=isThan16(results1)

            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q,w in enumerate(add_label):

                if (w==-1):
                    if (addLabel2[q] <=-1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if(addLabel2[q] <= -1):
                        continue
                    else:
                        if(addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if(add_label3[x1]>-1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])

            # addImage_proReal = label_tasks[i]["query"].squeeze_()[addImage_pro]
            #
            #
            # label_acc.append(calculate_accuracy(addLabel_pro, addImage_proReal))
            # label_num += len(addImage_pro)
            # than16+= isThan16(addLabel_pro)

            #proto_pred2 = protoPred(scores2, 1)
            # 添加类别不平衡
            # addImage_pro,addLabel_pro = class_balance(scores2,indices2,knn_trainLabel2,addImage_pro,addLabel_pro,self.k)


            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features,query_features[addImage_pro]),dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)
                addLabel_tensor2 = torch.tensor(addLabel_pro)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor),dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()

            # -------------------------------------------
            if self.support == 1:
                score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features[addImage_pro], support_features, label_tasks[i]["support"].squeeze_().cuda(),self.k-1, **kwargs)
            else:
                score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features[addImage_pro], support_features, label_tasks[i][
                                                                                           "support"].squeeze_().cuda(),
                                                                                       self.k, **kwargs)
            knn_trainLabel22 = label_tasks[i]["support"].squeeze_()

            b_tensor = torch.tensor(indices3)
            c_tensor = knn_trainLabel22[b_tensor]
            label_diff2 = torch.abs(c_tensor - addLabel_tensor2.unsqueeze(1).expand(-1, indices3.shape[1]))

            # 计算非零元素的数量
            non_zero_count2 = torch.sum(label_diff2 != 0, dim=1)
            # 计算损失并进行指数运算

            knn_loss2 = non_zero_count2.float() / self.k
            # 计算平均损失
            knn_loss2 = knn_loss2.mean().item()

            loss2 += (F.cross_entropy(score3, addLabel_tensor2.cuda())) + (knn_loss2)

            # ------------------------------------------------------------
            # addLabel_loss2 = torch.tensor(addLabel)
            # score3, indices3, knn_distances3, knn_pred3, scores3 = self.classifier(support_features, label_tasks[i][
            #     "support"].squeeze_().cuda(), query_features[addImage], support_features, label_tasks[i][
            #                                                                            "support"].squeeze_().cuda(),
            #                                                                        self.k, **kwargs)
            # knn_trainLabel22 = label_tasks[i]["support"].squeeze_()
            #
            # b_tensor = torch.tensor(indices3)
            # c_tensor = knn_trainLabel22[b_tensor]
            # label_diff2 = torch.abs(c_tensor - addLabel_loss2.unsqueeze(1).expand(-1, indices3.shape[1]))

            # 计算非零元素的数量
            # non_zero_count2 = torch.sum(label_diff2 != 0, dim=1)
            # # 计算损失并进行指数运算
            #
            # knn_loss2 = non_zero_count2.float() / self.k
            # # 计算平均损失
            # knn_loss2 = knn_loss2.mean().item()
            #
            # loss2 += (F.cross_entropy(score3, addLabel_loss2.cuda())) + (knn_loss2)

            #-------------------------------------------------------------------------

            score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,proto_support_labels,query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), self.k, **kwargs)


            knn_trainLabel = proto_support_labels.cpu()
            # knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"].squeeze_()), dim=0)
            # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,
                                                                                                     indices.shape[1]))
            # 计算非零元素的数量
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            # 计算损失并进行指数运算
            # knn_loss = torch.exp(non_zero_count.float() / self.k)
            knn_loss = non_zero_count.float() / self.k
            # 计算平均损失
            knn_loss = knn_loss.mean().item()

            # loss += 0.9*(F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda()))+0.1*(knn_loss)
            # loss += 0.9*(F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + 0.1*(knn_loss)
            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)

            # proto_confi = getprotoconfi(score,torch.squeeze(label_tasks[i]["query"]))  # 获得原型网络的置信度
            # knn_confi = getknnconfi(indices, label_tasks[i]["support"],torch.squeeze(label_tasks[i]["query"]), 5).cuda()  # 获得knn的置信度

            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)  # 获得knn的置信度


            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
                                 self.k)
            # new_pred = torch.where(0.7*proto_confi[0] > 0.3*(knn_confi), proto_pred[0], knn_st_pred.to(torch.long))
            new_pred = torch.where(self.lam * proto_confi[0] > (1-self.lam) * (knn_confi), proto_pred[0], knn_st_pred.to(torch.long))

            new_pred_idx = [i for i in range(75)]

            new_pred3 = new_pred.clone()
            new_pred_idx,new_pred2,updata_idx,proto_confi_stas,knnConfi_stas = class_balance(scores, indices, proto_support_labels, new_pred_idx, new_pred3,self.k)

            # statistic_balance(torch.exp(score[updata_idx]),knn_trainLabel[indices][updata_idx],new_pred[updata_idx],new_pred2[updata_idx],(label_tasks[i]["query"][updata_idx]).cuda(),proto_confi_stas[updata_idx],knnConfi_stas[updata_idx])

            # idx1_acc,idx2_acc,balance_sum = stastic_balance(new_pred[updata_idx],new_pred2[updata_idx],(label_tasks[i]["query"][updata_idx]).cuda())
            #
            # balance_acc1 +=idx1_acc
            # balance_acc2 += idx2_acc
            # balance_acc3 += balance_sum

            new_pred = new_pred2

            # new_pred = torch.where(proto_confi3 > knn_confi3, proto_pred[0], knn_st_pred.to(torch.long))


            #statistic2(knn_pred,proto_pred,label_tasks[i]["query"].squeeze_().cuda(),scores,knn_distances,knn_trainLabel[indices],pre_knn_confi,knn_confi,pre_proto_confi,proto_confi,addImage,addLabel)
            acc.append(
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            knn_acc.append(
                torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            proto_acc.append(torch.tensor(
                (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            label_total_num+=75
            # acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        model.train()
        # optimizer.zero_grad()
        loss2 /= batch_size
        loss2.backward()
        optimizer.step()
        model.eval()
        # all_data_combined = np.vstack(array)
        # df = pd.DataFrame(all_data_combined)
        # # 将DataFrame写入Excel表格
        # excel_file = 'statistic_balance.xlsx'
        # df.to_excel(excel_file, index=False)

        return loss, knn_acc, proto_acc, acc,label_acc1,label_num1,label_acc2,label_num2,label_acc,label_num,label_total_num,than16_1,than16_2,than16,balance_acc1,balance_acc2,balance_acc3


    def forward2(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        knn_acc = []
        proto_acc = []
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier2(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), label_tasks[i]["query"].squeeze_().cuda(),self.k, **kwargs)

            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = torch.cat((label_tasks[i]["support"].squeeze_().cuda(), label_tasks[i]["query"].squeeze_().cuda()), dim=0)
            addImage,addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)


            labelnum = self.train_way * self.query
            temp = self.support * self.train_way
            addLabel2 = addImages2(support_features, label_tasks[i]["support"].squeeze_().cuda(), query_features,
                                   labelnum, temp)

            addLabel2Index = []
            for addLabel2Index1, addLabel2Index2 in enumerate(addLabel2):
                if (addLabel2Index2 > -1):
                    addLabel2Index.append(addLabel2Index1)


            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q, w in enumerate(add_label):

                if (w == -1):
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        if (addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if (add_label3[x1] > -1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])

            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features, query_features[addImage_pro]), dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor), dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()


            score, indices, knn_distances, knn_pred, scores = self.classifier2(proto_support_images,proto_support_labels,query_features, support_features,
                                                                               label_tasks[i]["support"].squeeze_().cuda(),
                                                                               label_tasks[i]["query"].squeeze_().cuda(),
                                                                               self.k,**kwargs)
            # score, indices, knn_pred, scores, knn_confi_weight, proto_confi_weight = \
            #     self.classifier(query_features,support_features,label_tasks[i]["support"].squeeze_().cuda(),**kwargs)

            knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"]), dim=0)
            # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
            label_diff = torch.abs(knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1, indices.shape[1]))
            # 计算非零元素的数量
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            # 计算损失并进行指数运算
            # knn_loss = torch.exp(non_zero_count.float() / self.k)
            knn_loss = non_zero_count.float() / self.k
            # 计算平均损失
            knn_loss = knn_loss.mean().item()

            # loss += 0.9*(F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda()))+0.1*(knn_loss)
            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)
            # loss += 0.9 * (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + 0.1 * (knn_loss)

            # proto_confi = getprotoconfi(score,torch.squeeze(label_tasks[i]["query"]))  # 获得原型网络的置信度
            # knn_confi = getknnconfi(indices, label_tasks[i]["support"],torch.squeeze(label_tasks[i]["query"]), 5).cuda()  # 获得knn的置信度

            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))

            pre_proto_confi,proto_confi = getprotoconfi(scores,proto_pred)  # 获得原型网络的置信度
            pre_knn_confi,knn_confi = getknnconfi(indices, knn_trainLabel,knn_pred.cpu(), self.k)  # 获得knn的置信度

            # proto_confi2 = proto_confi.unsqueeze(1) * proto_confi_weight
            # proto_confi2_sum = torch.sum(proto_confi2,dim=1)
            # proto_confi3 = proto_confi2_sum/proto_confi_weight.size(0)
            #
            # knn_confi2 = knn_confi.unsqueeze(1) * knn_confi_weight
            # knn_confi2_sum = torch.sum(knn_confi2, dim=1)
            # knn_confi3 = knn_confi2_sum / knn_confi_weight.size(0)



            # 考虑了原型距离的knn预测结果
            knn_st_pred = knn_st(knn_distances,knn_trainLabel,indices,self.train_way*self.query,self.train_way,self.k)
            # new_pred = torch.where(proto_confi[0] > knn_confi, proto_pred[0], knn_st_pred.to(torch.long))
            new_pred = torch.where(self.lam * proto_confi[0] > (1 - self.lam) * (knn_confi), proto_pred[0],
                                   knn_st_pred.to(torch.long))

            #statistic(knn_pred,knn_st_pred,proto_pred,new_pred,label_tasks[i]["query"].squeeze_().cuda(),knn_confi,proto_confi,pre_knn_confi,pre_proto_confi)
            acc.append(torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item())*100)
            knn_acc.append(torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item())*100)
            proto_acc.append(torch.tensor(
                (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)

            #acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        return loss, knn_acc,proto_acc,acc

    # def forward3(self,img_tasks,label_tasks, *args, **kwargs):
    #     batch_size = len(img_tasks)
    #     loss = 0.
    #     knn_acc = []
    #     proto_acc = []
    #     acc = []
    #     for i, img_task in enumerate(img_tasks):
    #         support_features = self.backbone(img_task["support"].squeeze_().cuda())
    #         query_features = self.backbone(img_task["query"].squeeze_().cuda())
    #
    #
    #         score, indices, knn_distances, knn_pred, scores = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
    #                                 label_tasks[i]["support"].squeeze_().cuda(), self.k, **kwargs)
    #
    #         knn_trainLabel = label_tasks[i]["support"].cpu()
    #         # knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"].squeeze_()), dim=0)
    #         # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
    #         label_diff = torch.abs(
    #             knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,indices.shape[1]))
    #         # 计算非零元素的数量
    #         non_zero_count = torch.sum(label_diff != 0, dim=1)
    #         # 计算损失并进行指数运算
    #         # knn_loss = torch.exp(non_zero_count.float() / self.k)
    #         knn_loss = non_zero_count.float() / self.k
    #         # 计算平均损失
    #         knn_loss = knn_loss.mean().item()
    #
    #
    #         loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)
    #
    #         proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["query"]))
    #
    #         pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
    #         pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)  # 获得knn的置信度
    #
    #
    #
    #         knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
    #                              self.k)
    #         new_pred = torch.where(0.63*proto_confi[0] > (knn_confi)*0.37, proto_pred[0], knn_st_pred.to(torch.long))
    #
    #
    #
    #         acc.append(
    #             torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
    #         knn_acc.append(
    #             torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
    #         proto_acc.append(torch.tensor(
    #             (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
    #
    #
    #     loss /= batch_size
    #     return loss, knn_acc, proto_acc, acc
    def forward3(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        knn_acc = []
        proto_acc = []
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())

            if self.support == 1:
                score2,indices2,knn_distances2,knn_pred2,scores2 = self.classifier(support_features,label_tasks[i]["support"].squeeze_().cuda(),query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), self.k-1, **kwargs)
            else:
                score2, indices2, knn_distances2, knn_pred2, scores2 = self.classifier(support_features, label_tasks[i][
                    "support"].squeeze_().cuda(), query_features, support_features,label_tasks[i]["support"].squeeze_().cuda(),self.k, **kwargs)
            proto_pred2 = protoPred(score2, torch.squeeze(label_tasks[i]["support"]))
            knn_trainLabel2 = label_tasks[i]["support"].squeeze_().cuda()
            addImage, addLabel = addImages(proto_pred2,knn_pred2,indices2,knn_trainLabel2)

            labelnum = self.train_way * self.query
            temp = self.support * self.train_way
            addLabel2 = addImages2(support_features, label_tasks[i]["support"].squeeze_().cuda(), query_features,
                                   labelnum, temp)

            addLabel2Index = []
            for addLabel2Index1, addLabel2Index2 in enumerate(addLabel2):
                if (addLabel2Index2 > -1):
                    addLabel2Index.append(addLabel2Index1)

            add_label = [-1 for _ in range(labelnum)]

            for i5 in range(len(addImage)):
                add_label[addImage[i5]] = addLabel[i5].item()

            add_label3 = add_label.copy()
            for q, w in enumerate(add_label):

                if (w == -1):
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        add_label3[q] = addLabel2[q]

                else:
                    if (addLabel2[q] <= -1):
                        continue
                    else:
                        if (addLabel2[q] == w):
                            continue
                        else:
                            add_label3[q] = -2

            addImage_pro = []
            addLabel_pro = []

            for x1 in range(len(add_label3)):
                if (add_label3[x1] > -1):
                    addImage_pro.append(x1)
                    addLabel_pro.append(add_label3[x1])

            if len(addImage_pro) != 0:

                proto_support_images = torch.cat((support_features, query_features[addImage_pro]), dim=0)

                addLabel_tensor = torch.tensor(addLabel_pro)
                addLabel_tensor2 = torch.tensor(addLabel_pro)
                proto_support_labels = torch.cat((label_tasks[i]["support"], addLabel_tensor), dim=0).cuda()
            else:
                proto_support_images = support_features
                proto_support_labels = label_tasks[i]["support"].squeeze_().cuda()

            # if len(addImage) == 0:
            #     score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,proto_support_labels,query_features, support_features,
            #                         label_tasks[i]["support"].squeeze_().cuda(), self.k-1, **kwargs)
            # else:
            #     score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,
            #                                                                       proto_support_labels, query_features,
            #                                                                       support_features,
            #                                                                       label_tasks[i][
            #                                                                           "support"].squeeze_().cuda(),
            #                                                                       self.k, **kwargs)

            score, indices, knn_distances, knn_pred, scores = self.classifier(proto_support_images,
                                                                              proto_support_labels, query_features,
                                                                              support_features,
                                                                              label_tasks[i][
                                                                                  "support"].squeeze_().cuda(),
                                                                              self.k, **kwargs)

            knn_trainLabel = proto_support_labels.cpu()
            # knn_trainLabel = torch.cat((label_tasks[i]["support"], label_tasks[i]["query"].squeeze_()), dim=0)
            # 计算标签差异值并求绝对值---------------这行有问题，label_tasks[i]["query"]和原来代码里的不一样
            label_diff = torch.abs(
                knn_trainLabel[indices] - torch.squeeze(label_tasks[i]["query"]).unsqueeze(1).expand(-1,
                                                                                                     indices.shape[1]))
            # 计算非零元素的数量
            non_zero_count = torch.sum(label_diff != 0, dim=1)
            # 计算损失并进行指数运算
            # knn_loss = torch.exp(non_zero_count.float() / self.k)
            knn_loss = non_zero_count.float() / self.k
            # 计算平均损失
            knn_loss = knn_loss.mean().item()

            # loss += 0.9*(F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda()))+0.1*(knn_loss)
            # loss += 0.9*(F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + 0.1*(knn_loss)
            loss += (F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())) + (knn_loss)

            # proto_confi = getprotoconfi(score,torch.squeeze(label_tasks[i]["query"]))  # 获得原型网络的置信度
            # knn_confi = getknnconfi(indices, label_tasks[i]["support"],torch.squeeze(label_tasks[i]["query"]), 5).cuda()  # 获得knn的置信度

            proto_pred = protoPred(score, torch.squeeze(label_tasks[i]["support"]))

            pre_proto_confi, proto_confi = getprotoconfi(scores, proto_pred)  # 获得原型网络的置信度
            pre_knn_confi, knn_confi = getknnconfi(indices, knn_trainLabel, knn_pred.cpu(), self.k)  # 获得knn的置信度



            knn_st_pred = knn_st(knn_distances, knn_trainLabel, indices, self.train_way * self.query, self.train_way,
                                 self.k)
            # new_pred = torch.where(0.63*proto_confi[0] > (knn_confi)*0.37, proto_pred[0], knn_st_pred.to(torch.long))

            new_pred = torch.where(self.lam * proto_confi[0] > (1 - self.lam) * (knn_confi), proto_pred[0],
                                   knn_st_pred.to(torch.long))



            #statistic2(knn_pred,proto_pred,label_tasks[i]["query"].squeeze_().cuda(),scores,knn_distances,knn_trainLabel[indices],pre_knn_confi,knn_confi,pre_proto_confi,proto_confi,addImage,addLabel)
            acc.append(
                torch.tensor((new_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            knn_acc.append(
                torch.tensor((knn_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)
            proto_acc.append(torch.tensor(
                (proto_pred == torch.squeeze(label_tasks[i]["query"]).cuda()).float().mean().item()) * 100)

            # acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        return loss, knn_acc, proto_acc, acc

    def train_forward(self, img_tasks,label_tasks, *args, **kwargs):
        # return self(img_tasks, label_tasks, *args, **kwargs)
        return self.forward2(img_tasks, label_tasks, *args, **kwargs)

    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):

        return self.forward3(img_tasks, label_tasks, *args, **kwargs)

    def test_forward(self, img_tasks,label_tasks, *args, model,optimizer,step,**kwargs):
        return self(img_tasks, label_tasks, *args, model=model,optimizer=optimizer,step = step, **kwargs)

def get_model(config):
    return EpisodicTraining(config)
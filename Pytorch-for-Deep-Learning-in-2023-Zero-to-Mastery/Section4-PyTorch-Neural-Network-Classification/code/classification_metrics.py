import torch

# 귀찮아짐. 알잖아 어케하는지.


class ClassificationMetrics(object):
    def __init__(self, num_classes, ground_truth, prediction) -> None:
        '''
        Precision, Accuracy, Recall, F1-Score, Confusion Matrix를 구하기 위한 클래스.

        Args:
            num_classes (int): 클래스 수
            ground_truth (torch.Tensor): target label
            prediction (torch.Tensor): Model's prediction

        Example:
            ```
            metrics = ClassificationMetrics(num_classes=4,
                                            ground_truth=y_test,
                                            prediction=y_pred)
            ```
        '''

        if ground_truth.shape != prediction.shape:
            raise 'GT와 Pred는 동일한 형태를 가져아합니다.'

        total_samples = len(ground_truth)
        self.tp = torch.argwhere(ground_truth == prediction)

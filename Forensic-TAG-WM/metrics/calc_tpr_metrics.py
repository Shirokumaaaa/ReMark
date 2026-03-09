from scipy.special import betainc

class TPRMetrics:
    def __init__(self, wm_len=256, user_number=1e6, fpr=1e-6):
        self.wm_len = wm_len
        self.user_number = user_number
        self.fpr = fpr
        
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None
        for i in range(self.wm_len):
            fpr_onebit = betainc(i+1, self.wm_len - i, 0.5)
            fpr_bits = betainc(i+1, self.wm_len - i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.wm_len
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.wm_len
                
    def update_tp_count(self, accuracy):
        if accuracy >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count + 1
        if accuracy >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1

    def get_tpr(self, sample_num):
        tpr_detection = self.tp_onebit_count / sample_num
        tpr_traceability = self.tp_bits_count / sample_num
        return tpr_detection, tpr_traceability
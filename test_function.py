# -*- coding: utf-8 -*-
"""


@author: Lucas
"""

import sys
import pandas as pd
import unittest
import numpy as np
from module_function import describe, check_value_error, continu_mod




class Testfunct(unittest.TestCase):
    def setUp(self):
        t1 = pd.core.series.Series([12,17,6,15,18,10,9,19,12,1,16,15,2,15,-3,4])
        t2 = pd.core.series.Series(["male","male","female","female","male","male","female","female","male","male","female","female","male","male","female","female"])
        t3 = pd.core.series.Series(["Q","S","C","Q","C","Q","C","S","C","C","C","Q","C","Q","S","S"])
        self.df_continu_mod_test = pd.concat((t1,t2,t3), axis= 1)
        self.df_continu_mod_test.columns=["Price","Sex","Embarked"]

#        self.data = pd.DataFrame(np.array([[12,"male", "Q"], [17,"male", "S"], [6,"female", "C"], [15,"female", "Q"],
#                                    [18,"male", "C"], [10,"male", "Q"], [9,"female", "C"], [19,"female", "S"],
#                                    [12,"male", "C"], [1,"male", "C"], [16,"female", "C"], [15,"female", "Q"],
#                                    [2,"male", "C"], [15,"male", "Q"], [3,"female", "S"], [4,"female", "S"],
#                                    ]), columns=['Price','Sex', 'Embarked'])
        self.continu_mod_test = {"cont_mod0" : [12,17,18,10,12,1,2,15],
                                "cont_mod1" : [6,15,9,19,16,15,-3,4],
                                "cont_mod2" : [],
                                "cont_mod3" : [],
                                "cont_mod4" : [],
                                "cont_mod5" : [],
                                "cont_mod6" : [],
                                "cont_mod7" : [],
                                "cont_mod8" : [],
                                "cont_mod9" : []
                                }
#        self.describe_df = pd.DataFrame(np.array([[5,"T",12,np.nan,"F","A+"],[12,"A",np.nan,9,"F","O+"],[16,"G",np.nan,16,"M","B+"],
#                                         [17,"G",np.nan,2,"M","O+"],[6,"G",np.nan,3,"M","A+"],[12,"",19,1,"F","B+"]]),
#                                         columns=["Price","gene","age","quali","Sex","Blood"])
        
        self.check_value_error_test = (1, [14])

#    def test_describe(self):
#       self.assertListEqual([1,1,np.array((4,5,6))],[1,1,np.array((4,5,6))])

    def test_check_value_error(self):
        self.assertTupleEqual(check_value_error(self.df_continu_mod_test["Price"],0,100),self.check_value_error_test)

    def test_continu_mod(self):
        self.assertDictEqual(continu_mod(self.df_continu_mod_test,"Price","Sex","male","female"),self.continu_mod_test)
if __name__ == '__main__':
    unittest.main()
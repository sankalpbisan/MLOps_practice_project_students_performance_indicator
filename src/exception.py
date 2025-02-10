from src.logger import logging
import sys

def error_message_detail(error,error_detail:sys):
    '''
    This function is designed to explicitly handle exception
    '''
    _,_,exc_tb = error_detail.exc_info() #the 1st and 2nd variable are not much helpful
        # 3rd variable stores all the details related to error,
        # like what the error is and on which it has occurred etc.....
    file_name =exc_tb.tb_frame.f_code.co_filename
    # This gives the filename
    error_message = "Error occurred in python file name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error))

    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message

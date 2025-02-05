import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    error_message="Error occured in Python script name [{0}] at line number [{1}] with error message [{2}]".format(exc_tb.tb_frame.f_code.co_filename,exc_tb.tb_lineno,error)
    return error_message
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    def __str__(self):
        return self.error_message
    
    

"""

PS-87 placeholder.

Implementation will be added in the relevant development task.

"""



from datetime import datetime

from pydantic import BaseModel, EmailStr





class UserResponse(BaseModel):

    id: str

    name: str

    email: str

    created_at: datetime



    class Config:

        from_attributes = True





class UserSignup(BaseModel):

    name: str

    email: EmailStr

    password: str





class UserLogin(BaseModel):

    email: EmailStr

    password: str





class Token(BaseModel):

    access_token: str

    token_type: str = "bearer"
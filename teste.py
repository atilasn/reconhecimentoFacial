import  numpy  as  np
import  cv2

cap  =  cv2 . VideoCapture ("C:/Temp/teste.avi")

while ( cap . isOpened ()):
    ret ,  frame  =  cap . read ()

    cinza  =  cv2 . cvtColor ( frame ,  cv2 . COLOR_BGR2GRAY )

    cv2 . imshow ( 'frame' , cinza )
    if  cv2 . waitKey ( 1 )  &  0xFF  ==  ord ( 'q' ):
        break

cap . release ()
cv2 . destroyAllWindows ()
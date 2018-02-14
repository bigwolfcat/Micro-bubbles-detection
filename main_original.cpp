#include <iostream>
//#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "../package_bgs/jmo/MultiLayerBGS.h"

#pragma omp parallel for

//*** Functions declaration ***
IplImage* img_resize(IplImage* src_img, int new_width,int new_height);
double countWhiteArea (cv::Mat image);
double countWhiteAreaPerc (cv::Mat image);
void imageErosion(cv::Mat im_in, cv::Mat im_out, int erosion_type, int erosion_size);


int main(int argc, char **argv)
{
    std::cout<<"\n*****************************************************************"<<std::endl;
    std::cout<<"                         Bubbles distribution 1.0\n            Author: Michele Svanera, November 2014"<<std::endl;
    std::cout<<"*****************************************************************\n"<<std::endl;
    
    if(argc!=3){
        std::cout<<"Usage: bgs [video.mp4] [out.txt]"<<std::endl;
        return 1;
    }
    
    CvCapture *capture = 0;
    capture = cvCaptureFromFile(argv[1]);
    
    std::cout<<"Video Name: "<<argv[1]<<std::endl;
    
    if(!capture) {
        std::cerr << "Cannot open initialize video file!" << std::endl;
        return 1;
    }
    
    std::vector<int> vector_of_areas;
    int bubbles_found = 0;
    std::ofstream myfile_txt (argv[2]);
    
    if (!myfile_txt.is_open()){
        std::cout << "Unable to open file" << std::endl << std::endl;
        return -1;
    }
    
    std::cout<<"TXT out Name: "<<argv[2]<<std::endl<<std::endl;
    
    IplImage *frame = cvQueryFrame(capture);
    MultiLayerBGS* bgs = new MultiLayerBGS;
    
    int first_frame = 0;
    double fps = 0, fpsAdd = 0, t = 0, q = 0;
    
    while(1)
    {
        bubbles_found = 0;
        t = (double)cvGetTickCount();
        frame = cvQueryFrame(capture);
        
        if(!frame) break;
        
        //*** frame adjustment ***
        //frame = img_resize(frame, frame->width/4*3,frame->height/4*3);
        cvSetImageROI(frame, cvRect(0, 50, frame->width, frame->height-100));
        IplImage *frame_tmp = cvCreateImage(cvGetSize(frame), frame->depth, frame->nChannels);
        cvCopy(frame, frame_tmp, NULL);
        cvResetImageROI(frame);
        frame = cvCloneImage(frame_tmp);
        cvReleaseImage(&frame_tmp);
        
        //*** Background subtraction ***
        cv::Mat img_input(frame,true);
        cv::Mat img_mask;
        cv::Mat img_out;
        bgs->process(img_input,img_out,img_mask);
        if(img_out.rows==0 || img_out.cols==0 || first_frame<=5)
        {
            first_frame++;
            continue;
        }
        
        
        if (img_out.channels() != 1) //if 3-channels image.
        {
            cv::Mat im_gray = cv::Mat::zeros(img_out.cols,img_out.rows, CV_8U);
            cvtColor(img_out, im_gray, CV_RGB2GRAY);
            img_out = im_gray.clone();
            im_gray.release();
        }
        
        cv::Mat img_out_bgs_clean = cv::Mat::zeros(img_out.cols,img_out.rows, CV_8U);
        img_out_bgs_clean = img_out.clone();
        
        //QUA SI POTREBBE INSERIRE DEL CODICE PER FARE MULTI-THREADING, UN THREAD OGNI IMMAGINI PRESA, IN MODO DA ACCELLERARE IL TUTTO. DA INSERIRE TUTTO IL CODICE DA QUA FINO ALLA FINE DEL CICLO WHILE. POI IL THREAD SALVA DA QUALCHE PARTE LE DIMENSIONI TROVATE. UN THREAD (O DEL CODICE NORMALE) FINALE DOVRA' POI CONTEGGIARE ATTRAVERSO TUTTI I FRAME E CREARE L'HIST FINALE.
        
        //*** BGS result elaboration ***
        equalizeHist( img_out, img_out );
        threshold(img_out, img_out, (int)(255*3/4), 255, CV_THRESH_BINARY);
        
        //*** findContours and rectangles ***
        cv::Mat img_out_contours = img_out.clone();
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        
        findContours(img_out_contours, contours, hierarchy, CV_RETR_TREE,
                     CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        
        std::vector<std::vector<cv::Point> > contours_poly(contours.size());
        std::vector<cv::Rect> boundRect(contours.size());
        
        for (int i = 0; i < contours.size(); i++) {
            approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(cv::Mat(contours_poly[i]));
        }
        
        
        //*** Check if I have a bubble 'i' inside another 'j' ***
        //'i' is the small one: check that its center is not inside the rectangle j.
        for (int i = 0; i < boundRect.size(); i++) {
            for (int j = 0; j < boundRect.size(); j++) {
                if(i==j) continue;  //Check if i!=j
                cv::Point center_of_rect = cv::Point(boundRect[i].x+boundRect[i].width/2,boundRect[i].y+boundRect[i].height/2);
                if(boundRect[j].contains(center_of_rect) & (boundRect[i].area()<boundRect[j].area())){
                    int plus_factor = 1;
                    cv::Rect boundRect_plus_pl_fac (boundRect[i].x-plus_factor>0 ? boundRect[i].x-plus_factor : boundRect[i].x, boundRect[i].y-plus_factor>0 ? boundRect[i].y-plus_factor : boundRect[i].y, boundRect[i].width+plus_factor*2+boundRect[i].x<img_out.cols ? boundRect[i].width+plus_factor*2 : boundRect[i].width, boundRect[i].height+plus_factor*2+boundRect[i].x<img_out.rows? boundRect[i].height+plus_factor*2 : boundRect[i].height);
                    rectangle(img_out,boundRect_plus_pl_fac.tl(), boundRect_plus_pl_fac.br(), cv::Scalar(255, 255, 255), -1, 8, 0);
                    boundRect.erase(boundRect.begin()+(i));
                    i--;
                }
            }
            //Delete rectangles with area less than 10 pixels.
            if(boundRect[i].area()<10){
                boundRect.erase(boundRect.begin()+(i));
                i--;
            }
        }
        
        //*** One iteration for every agglomerate of bubbles found ***
        //std::cout<<(int)boundRect.size()<<" agglomerates found."<<std::endl;
        for (int i = 0; i < boundRect.size(); i++) {
            
            cv::Mat bubble_agglomerated = img_out(boundRect[i]);
            
            //*** If white >= 70% of_total_area, then it's a bubble and I count the area ***
            if((int)countWhiteAreaPerc(bubble_agglomerated)>=70){
                vector_of_areas.push_back(countWhiteArea(bubble_agglomerated));
                bubbles_found ++;
                continue;
            }
            
            cv::Mat bubble_agglomerated_from_input = img_out_bgs_clean(boundRect[i]);
            //cv::Mat to_visualize = bubble_agglomerated_from_input.clone();
            if (bubble_agglomerated_from_input.channels() != 1) //Se sono 3-channels image.
            {
                cv::Mat im_gray = cv::Mat::zeros(bubble_agglomerated_from_input.cols,bubble_agglomerated_from_input.rows, CV_8U);
                cvtColor(bubble_agglomerated_from_input, im_gray, CV_RGB2GRAY);
                bubble_agglomerated_from_input = im_gray.clone();
                im_gray.release();
            }
            
            int multiplier = 10;
            cv::resize(bubble_agglomerated_from_input,bubble_agglomerated_from_input,cv::Size(bubble_agglomerated_from_input.cols*multiplier,bubble_agglomerated_from_input.rows*multiplier));
            //cv::resize(to_visualize,to_visualize,cv::Size(to_visualize.cols*multiplier,to_visualize.rows*multiplier));
            
            
            cv::Mat bubble_agglomerated_from_input_equalized;
            bubble_agglomerated_from_input_equalized = bubble_agglomerated_from_input.clone();
            equalizeHist(bubble_agglomerated_from_input, bubble_agglomerated_from_input_equalized);
            
            
            //GaussianBlur( bubble_agglomerated_from_input, bubble_agglomerated_from_input, Size(9, 9), 2, 2 );
            std::vector<cv::Vec3f> circles;
            
            HoughCircles( bubble_agglomerated_from_input_equalized, circles, CV_HOUGH_GRADIENT, 1, 1, 30, 105, 0, 0 );
            
            cv::Mat full_circles_image = cv::Mat::zeros(bubble_agglomerated_from_input_equalized.rows, bubble_agglomerated_from_input_equalized.cols, CV_8U);
            
            for( size_t i = 0; i < circles.size(); i++ )
            {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                circle( full_circles_image, center, 3, cv::Scalar(255,255,255), -1, 8, 0 );
                circle( full_circles_image, center, radius, cv::Scalar(255,255,255), -1, 8, 0 );
            }
            
            //*** Some of erosion in order to split (only) very close circles ***
            imageErosion(full_circles_image,full_circles_image, cv::MORPH_ELLIPSE, 11);
            
            //cv::imshow("full_circles_image", full_circles_image);
            //cv::imshow("aggl_from_in_equalized", bubble_agglomerated_from_input_equalized);
            
            
            //*** findContours and rectangles ***
            cv::Mat full_circles_image_contours = full_circles_image.clone();
            std::vector<std::vector<cv::Point> > circles_contours;
            std::vector<cv::Vec4i> circles_hierarchy;
            
            findContours(full_circles_image_contours, circles_contours, circles_hierarchy, CV_RETR_TREE,
                         CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
            
            std::vector<std::vector<cv::Point> > circles_contours_poly(circles_contours.size());
            std::vector<cv::Rect> circles_boundRect(circles_contours.size());
            
            for (int i = 0; i < circles_contours.size(); i++) {
                approxPolyDP(cv::Mat(circles_contours[i]), circles_contours_poly[i], 3, true);
                circles_boundRect[i] = boundingRect(cv::Mat(circles_contours_poly[i]));
            }
            
            for (int i = 0; i < circles_boundRect.size(); i++) {
                //rectangle(to_visualize, circles_boundRect[i].tl(), circles_boundRect[i].br(), cv::Scalar(255, 0, 0), 1, 8, 0);
                cv::Mat last_split_bubbles = full_circles_image(circles_boundRect[i]);
                
                //*** If white >= 70% of_total_area, then it's a bubble and I count the area ***
                if((int)countWhiteAreaPerc(last_split_bubbles)>=70){
                    vector_of_areas.push_back((int)countWhiteArea(last_split_bubbles)/(multiplier*multiplier));
                    bubbles_found++;
                }
                last_split_bubbles.release();
            }
            
            bubble_agglomerated.release();
            bubble_agglomerated_from_input.release();
            bubble_agglomerated_from_input_equalized.release();
            full_circles_image.release();
            full_circles_image_contours.release();
            
            /*
             cv::imshow("to_visualize", to_visualize);
             cv::imshow("aggl_from_in_equalized", bubble_agglomerated_from_input_equalized);
             cvWaitKey(0);*/
        }
        std::cout<<"Bubbles found in frame "<< "n : " <<(int)bubbles_found<<"."<<std::endl;
        
        
        
        
        
        
        //Stima del tempo
        t = (double)cvGetTickCount() - t;
        t = t/((double)cvGetTickFrequency()*1000.) ;
        fpsAdd +=t;
        q++;
        if((fpsAdd>=1000))
        {
            fps = ((double)(q*1000)/fpsAdd);
            q=0;
            fpsAdd=0;
            //printf("FPS: %.1f\n",fps);
        }
        
        //Clear memory
        img_out.release();
        img_input.release();
        img_mask.release();
        img_out_bgs_clean.release();
        
        //*** Save in a txt file all the values obtained from the bubbles and saved in vector_of_areas ***
        if (myfile_txt.is_open())
        {
            for (std::vector<int>::iterator it = vector_of_areas.begin() ; it != vector_of_areas.end(); ++it)
                myfile_txt << sqrt((double)((*it)/M_PI)) << std::endl; //I save radius! Not area!
            
            vector_of_areas.clear();
        }
        else{
            std::cout << "Unable to open file" << std::endl << std::endl;
            return -1;
        }
        
    }//Fine while
    
    
    myfile_txt.close();
    std::cout<<" done!"<<std::endl;
    
    
    //*** Clear memory ***
    
    delete bgs;
    
    cvDestroyAllWindows();
    cvReleaseImage(&frame);
    cvReleaseCapture(&capture);
    
    return 0;
}

IplImage* img_resize(IplImage* src_img, int new_width,int new_height)
{
    IplImage* des_img;
    des_img=cvCreateImage(cvSize(new_width,new_height),src_img->depth,src_img->nChannels);
    cvResize(src_img,des_img,CV_INTER_LINEAR);
    return des_img;
}


double countWhiteArea (cv::Mat image)
{
    return (sum(image)[0]/255);
}

double countWhiteAreaPerc (cv::Mat image)
{
    return (100*(sum(image)[0]/255)/(image.cols*image.rows));
}

void imageErosion(cv::Mat im_in, cv::Mat im_out, int erosion_type, int erosion_size)
{
    cv::Mat element = getStructuringElement( erosion_type,
                                            cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                            cv::Point( erosion_size, erosion_size ) );
    
    /// Apply the erosion operation
    erode( im_in, im_out, element );
}


//       cv::namedWindow("threshold", CV_WINDOW_NORMAL );

//cv::resize(img_input,img_input,cv::Size(320,240));

//	imwrite( "./img_input.jpg", img_input );




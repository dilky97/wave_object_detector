import logging
from h2o_wave import Q, ui, app, main

from src.object_detector import ObjectDetector

@app('/')
async def serve(q:Q):
    """
    Entry to the app
    :param q: Query argument from the H2O Wave server
    :return: None
    
    """
    print(q.args)
    await initialize_app(q)
    await q.page.save()


async def display_detections(q):
    """
    Display the image with bounding boxes for detected objects
    :param q: Query argument from the H2O Wave server 
    :return None
    """
    image_path = "./data/"+ q.client.uploaded_path[2:]

    # If user has not picked what objects to display, display all the detections
    if not q.client.picked_classes:
        q.page['content'] = ui.form_card(
            box = ui.box('content'),
            title = "Object Detector",
            items = [
                ui.text("No objects picked. Displaying all"),
                ui.picker(name='object_picker', label='Pick objects to detect', width = '50%', trigger=True,
                    choices=[ui.choice(thing, thing) for thing in q.app.object_detector.classes_names] , values=[]),
            ],
        )

        q.page['uploaded_image'] = ui.image_card(
            box = ui.box('content', width='900px', height='550px'),
            title = '',
            type = 'png',    
            image = q.app.object_detector.detect_objects(image_path,q.client.picked_classes),
            # path=image_path
        )

    # Display the selected object types
    else:
        q.page['content'] = ui.form_card(
                box = ui.box('content'),
                title = "Object Detector",
                items = [
                    ui.picker(name='object_picker', label='Pick objects to detect', width = '50%', trigger=True,
                        choices=[ui.choice(thing, thing) for thing in q.app.object_detector.classes_names] , values=[]),

                ],
            )

        q.page['uploaded_image'] = ui.image_card(
            box = ui.box('content', width='900px', height='550px'),
            title = '',
            type = 'png',    
            image = q.app.object_detector.detect_objects(image_path,q.client.picked_classes),
            # path=image_path
        )
    
    q.client.picked_classes = []
    await q.page.save()


async def set_sample_path(q):
    """
    Path handler for sample images.
    
    """
    option = q.client.dropdown_option
    if option == 'nyc':
        sample_image_path = './static/nyc.jpg'
        q.client.uploaded_path, = await q.site.upload([sample_image_path])
        await display_detections(q)
    
    if option == 'horserace':
        sample_image_path = './static/horserace.jpg'
        q.client.uploaded_path, = await q.site.upload([sample_image_path])
        await display_detections(q)

    
async def initialize_app(q:Q):
    """
    Initialize the app prior to first usage
    :param q: Query argument from the H2O Wave server
    :return: None
    """
    logging.info("Initializing the app.")

    if not q.app.initialized:
        q.app.initialized = True
        q.app.object_detector = ObjectDetector()
        q.client.dropdown_option = 'horserace'
        q.client.picked_classes = []
        q.client.uploaded_path = ''

        logging.info("App initialization complete")

    if not q.client.initialized:
        
        q.client.initialized = True

        #UI Layout
        q.page['meta'] = ui.meta_card(box='', layouts=[    
            ui.layout(        
                breakpoint='xl', 
                width = '1200px',
                # max_width='1200px',
                max_height = '800px',        
                zones=[            
                    ui.zone('header'),      
                    ui.zone('body', direction=ui.ZoneDirection.ROW, zones=[                
                        ui.zone('sidebar', size='25%'),
                        ui.zone('content', size='75%'),                            
                        ]),           
                    ui.zone('footer'),        
                ]
            )
        ])

        q.page['header'] = ui.header_card(box='header', title='Object Detector', subtitle='Using Detectron2')

        q.page['sidebar'] = ui.form_card(
            box = 'sidebar',
            title ="Upload an Image",
            items =[
                ui.file_upload(name='file_upload', label='Upload an Image', multiple=False, file_extensions=['jpg', 'png','mp4']),
                ui.separator(label=''),
                ui.dropdown(name='dropdown_images', label='Choose a sample image', value=q.client.dropdown_option, choices=[
                    ui.choice(name="nyc", label="NYC.jpg"),
                    ui.choice(name="horserace", label="Horse Race.jpg")
                ]),
                ui.button(name='show_image', label='Use the sample', primary=True),
            ],
        )

        q.page['content'] = ui.section_card(    
            box='content',    
            title='Object Detector',    
            subtitle='Show Results'
        )


        q.page['footer'] = ui.footer_card(box='footer', caption="Made with H2O Wave. Images downloaded from pexels.com")


    # Runs when user uploads a custom image
    if q.args.file_upload:
        q.client.uploaded_path = q.args.file_upload[0]
        await display_detections(q)
    
    # Runs when user chooses a sample image from the dropdown
    if q.args.dropdown_images:
        q.client.dropdown_option = q.args.dropdown_images

    if q.args.show_image:
        await set_sample_path(q)
    
    # Runs when user chooses which objects to detect
    if q.args.object_picker:
        q.client.picked_classes = q.args.object_picker
        await display_detections(q)

    



    



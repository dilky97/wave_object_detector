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


async def display_detections(q,image_path):

    image_path = "./data/"+ image_path[2:]

    q.page['uploaded_image'] = ui.image_card(
        box = ui.box('content', width='900px', height='500px'),
        # box = 'content',
        title = '',
        type = 'png',    
        image = q.app.object_detector.detect_objects(image_path),
        # path=image_path
    )
    await q.page.save()


async def set_sample_path(q):
    option = q.client.dropdown_option
    if  option == 'nyc':
        sample_image_path = './static/nyc.jpg'
        uploaded_path, = await q.site.upload([sample_image_path])
        await display_detections(q,uploaded_path)
    
    if option == 'horserace':
        sample_image_path = './static/horserace.jpg'
        uploaded_path, = await q.site.upload([sample_image_path])
        await display_detections(q,uploaded_path)

    
async def initialize_app(q:Q):
    if not q.app.initialized:
        q.app.initialized = True
        q.app.object_detector = ObjectDetector()
        q.client.dropdown_option = 'nyc'

        logging.info("App initialization complete")

    if not q.client.initialized:
        # setup app
        q.client.initialized = True

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
            box = ui.box(zone='sidebar'),
            title ="Upload an Image",
            items =[
                ui.file_upload(name='image_upload', label='', multiple=False, file_extensions=['jpg', 'png']),
                ui.dropdown(name='dropdown_images', label='Choose a sample image', value=q.client.dropdown_option, choices=[
                    ui.choice(name="nyc", label="NYC.jpg"),
                    ui.choice(name="horserace", label="Horse Race.jpg")
                ]),
                ui.button(name='show_image', label='Use the sample', primary=True),
            ],
        )

        q.page['content'] = ui.section_card(
            box = 'content',
            title = "Object Detector",
            subtitle = '',
            items=[
                # ui.toggle(name='search', label="A", value=True),
                # ui.dropdown(name='task', label='Choose the task', value='option0', choices=[
                #     ui.choice(name="object_detection", label="Object Detection"),
                #     ui.choice(name="instance_segmentation", label="Instance Segmentation")
                # ]),
                # ui.date_picker(name='target_date', label='', value='2020-12-25'),
            ],
        )

        q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')


    if q.args.image_upload:
        # await add_progress(q)
        uploaded_path = q.args.image_upload[0]
        # image_path = await q.site.download(uploaded, './temp/')
        print(uploaded_path)

        await display_detections(q,uploaded_path)
    
    if q.args.dropdown_images:
        q.client.dropdown_option = q.args.dropdown_images

    if q.args.show_image:
        await set_sample_path(q)




    



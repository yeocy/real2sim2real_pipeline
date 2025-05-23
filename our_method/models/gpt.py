import requests
import base64
import time
import os
import datetime
import json
import shutil

class GPT:
    """
    Simple interface for interacting with GPT-4O model
    """
    VERSIONS = {
        "4v": "gpt-4-vision-preview",
        "4o": "gpt-4o",
        "4o-mini": "gpt-4o-mini",
    }

    def __init__(
            self,
            api_key,
            version="4o",
            max_retries=3,
            log_dir_tail="",
    ):
        """
        Args:
            api_key (str): Key to use for querying GPT
            version (str): GPT version to use. Valid options are: {4o, 4o-mini, 4v}
            max_retries (int): The maximum number of retries to prompt GPT when receiving server error
        """
        self.api_key = api_key
        assert version in self.VERSIONS, f"Got invalid GPT version! Valid options are: {self.VERSIONS}, got: {version}"
        self.version = version
        self.max_retries = max_retries
        self.log_dir_tail = log_dir_tail

    def __call__(self, payload, verbose=False):
        """
        Queries GPT using the desired @prompt

        Args:
            payload (dict): Prompt payload to pass to GPT. This should be formatted properly, see
                https://platform.openai.com/docs/overview for details
            verbose (bool): Whether to be verbose as GPT is being queried

        Returns:
            None or str: Raw outputted GPT response if valid, else None
        """

        attempts = 0
        while attempts < self.max_retries:
            try:
                if verbose:
                    print(f"Querying GPT-{self.version} API...")

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.query_header, json=payload)
                response.raise_for_status()  # Raise an error for HTTP error responses
                response_data = response.json()

                if "choices" not in response_data.keys():
                    raise ValueError(f"Got error while querying GPT-{self.version} API! Response:\n\n{response.json()}")

                log_path = self.log_query_contents(request_msg=payload, request_img_path="", response=response_data, log_dir_tail=self.log_dir_tail)
                if verbose:
                    print(f"Finished querying GPT-{self.version}.")
                    print(f"GPT Query & Response Saved at {log_path}.")

                return response_data["choices"][0]["message"]["content"]
            
            except Exception as e:
                attempts += 1
                print(f"Error querying GPT-{self.version} API: {e}")
                if attempts < self.max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to query GPT-{self.version} API after {self.max_retries} attempts.")
                    return None

    @property
    def query_header(self):
        """
        Returns:
            dict: Relevant header to pass to all GPT queries
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def encode_image(self, image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            str: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def log_query_contents(self, request_msg, request_img_path, response, log_dir_base="logs", log_dir_tail=""):
        """
        현재 날짜와 시간을 파일명으로 사용하여 메시지를 로깅합니다.
        
        Parameters:
        request_msg (list): 로깅할 request 메시지 콘텐츠
        request_img_path (str): 로깅할 request 이미지 경로
        response (ChatCompletion): API 응답 객체
        log_dir_base (str): 로그 파일이 저장될 base 디렉토리 (기본값: "logs")
        
        Returns:
        str: 생성된 로그 파일의 경로
        """
        # 로그 디렉토리가 없으면 생성
        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)
        
        # 현재 날짜와 시간을 가져와서 파일명 형식으로 변환
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(log_dir_base, f"{timestamp}{log_dir_tail}")
        os.makedirs(log_dir)
        
        request_filename = 'query_contents_request.txt'
        response_filename = 'query_contents_response.txt'
        
        # 전체 파일 경로 생성
        request_file_path = os.path.join(log_dir, request_filename)
        response_file_path = os.path.join(log_dir, response_filename)
        
        # 파일에 request 메시지 작성
        with open(request_file_path, "w", encoding="utf-8") as f:
            json.dump(request_msg, f, indent=4)

        # 이미지 저장
        if request_img_path and os.path.exists(request_img_path):
            filename = os.path.basename(request_img_path)
            destination = os.path.join(log_dir, filename)

            # 이미지 복사
            shutil.copy(request_img_path, destination)
            print(f"이미지 저장 완료 ({destination})")

        # ChatCompletion 객체를 직렬화 가능한 딕셔너리로 변환
        # OpenAI 응답 객체일 경우
        if hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        # 또는 딕셔너리 속성 접근이 가능한 경우
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__
        elif type(response) == dict:
            response_dict = response
        # 다른 방법으로도 안되면 문자열로 저장
        else:
            response_dict = {'response_str': str(response)}
        # 파일에 response 메시지 작성
        with open(response_file_path, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=4)

        return log_dir

    def payload_get_object_caption(self, img_path):
        """
        Generates custom prompt payload for object captioning

        Args:
            img_path (str): Absolute path to image to caption

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_image = self.encode_image(img_path)

        prompting_text_system = "You are an expert in image captioning. " + \
                            "### Task Description ###\n\n" + \
                            "The user will give you an image, and please provide a list of all objects ([object1, object2, ...]) visible in the image. " + \
                            "For objects with visible and openable doors and drawers, please also return the number of doors (those with revolute joint, that can rotate around an axis) and drawers (those with prismatic joint, that can slide along a direction).\n\n" + \
                            "### Special Requirements ###\n\n" + \
                            "1. Treat each item as a single entity; avoid breaking down objects into their components. For instance, mention a wardrobe as one object instead of listing its doors and handles as separate items; " + \
                            "mention a pot plant/flower as a whole object instead of listing its vase separately.\n\n" + \
                            "2. When captioning, please do not include walls, floors, windows and any items hung from the ceiling in your answer, but please include objects installed or hung on walls.\n\n" + \
                            "3. When captioning, you can use broader categories. For instance, you can simply specify 'table' instead of 'short coffee table'.\n\n" + \
                            "4. Please caption all objects, even if some objects are closely placed, or an object is on top of another, or some objects are small compared to other objects. " + \
                            "However, don't come up with objects not in the image.\n\n" + \
                            "5. Please do not add 's' or 'es' suffices to countable nouns. For example, you should caption multiple apples as 'apple', not 'apples'.\n\n" + \
                            "6. When counting the number of doors and drawer, pay attention to the following:\n\n" + \
                            "(1). A child link cannot be a door and a drawer at the same time. When you are not sure if a child link is a door or a drawer, choose the most likely one.\n" + \
                            "(2). Please only count openable doors and drawers. Don't include objects with fixed and non-openable drawers/shelves/baskets (e.g., shelves/baskets/statis drawers of bookshelves, shelves, storage carts). For these objects, just give me the caption (e.g., bookshelf, shelf, storage cart).\n\n" + \
                            "Example output1: [banana, cabinet(3 doors & 3 drawers), chair]\n" + \
                            "Example output2: [wardrobe(2 doors), table, storage cart]\n" + \
                            "Example output3: [television, apple, shelf]\n" + \
                            "Example output4: [cabinet(8 drawers), desk, frying pan]\n\n\n"
        
        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        
        content_user = [
            {
                "type": "text",
                "text": "Now please provide a list of all objects ([object1, object2, ...]) visible in the image below.\n"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        object_caption_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 500
        }

        return object_caption_payload

    def payload_select_object_from_list(self, img_path, obj_list, bbox_img_path, nonproject_obj_img_path):
        """
        Generates custom prompt payload for selecting an object from a list of objects

        Args:
            img_path (str): Absolute path to image to infer object selection from
            obj_list (list of str): List of previously detected objects
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_image = self.encode_image(img_path)
        base64_target_obj_mask_image = self.encode_image(nonproject_obj_img_path)
        base64_target_obj_bbox_image = self.encode_image(bbox_img_path)

        prompting_text_system = "You are an expert in object captioning.\n\n" + \
                            "### Task Overview ###\n" + \
                            "The user will show you a scene, a target object in the scene highlighted by a bounding box, " + \
                            "and the mask of the target object in the scene where only the target object is shown in its original color and all other objects are masked as black pixles.\n\n" + \
                            "Next, the user will give you a list of object captions, each describing one or multiple objects in the scene. " + \
                            "Your task is to select the best caption from this list that most accurately describes the target object.\n\n" + \
                            "### Special Requirements ###\n\n" + \
                            "Please follow these guidelines when selecting your answer:\n\n" + \
                            "1. If multiple captions could be correct, choose the one that most accurately describes the target object.\n\n" + \
                            "2. Select a caption for the entire object. For instance, if the target object is a cabinet with doors, choose 'cabinet' instead of 'door.' " + \
                            "Similarly, if the object is a plant in a jar, choose 'plant' instead of 'jar'.\n\n" + \
                            "3. Focus on the target object.\n\n" + \
                            "(1) There may be occlusions or nearby objects included in the bounding box and the mask. " + \
                            "For example, if a bowl is in front of a cup, the bounding box and mask for the cup might contain part of the bowl (due to occlusion). " + \
                            "If some objects are too close to the target object, like an apple in a plate, then the bounding box and mask for the plate can include part of the apple (due to adjacency).\n" + \
                            "Ensure that you caption the intended object, not the occluding or adjacent one.\n\n" + \
                            "(2) In the object mask, the target object is shown in its original color, while all other objects are masked as black. " + \
                            "If multiple objects are on top or adjacent to the target object, the mask of the target object can contain a outline of other objects. " + \
                            "Please focus on the target object.\n\n" + \
                            "(3) Each bounding box and mask refers to only one target object, and the box usually centers at the target object. " + \
                            "You can use these two principles to help infer the target object under occlusion and adjacency.\n\n" + \
                            "4. If the target object is heavily occluded, you can use your common sense to infer the most likely caption of the target object. " + \
                            "For example, if multiple fruits are in the plate, the plate might be heavily occluded. Suppose the list of object captions contain 'fork' and 'plate', " + \
                            "based on common sense, you can infer the target object is more possible to be a plate, because it is strange that fruits are 'in' or 'on' a fork.\n\n" + \
                            "Similar situations happen when multiple objects are on top of in the target object, causing occusion to the target object. When you give your answer, please make sure it is not counter-intuitive.\n\n" + \
                            "5. Only select a caption from the provided list. Do not create any new caption that is not in the list.\n\n" + \
                            "6. Provide only the most appropriate caption, without any explanation.\n\n" + \
                            'Example output: banana\n\n'

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": "The above image shows a scene. " + \
                        "The following image shows the mask of the target object, where only the target object is shown in its original color with all other objects masked out as black."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_target_obj_mask_image}"
                }
            },
            {
                "type": "text",
                "text": "The following image shows the same object highlighted by a bounding box." 
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_target_obj_bbox_image}"
                }
            },
            {
                "type": "text",
                "text": f"The list of object captions of the scene is: {obj_list}.\n\n." + \
                    "Now please select the best caption from the list for the target object."
            }
        ]

        object_selection_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",  
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }

        return object_selection_payload

    def payload_count_drawer_door(self, caption, bbox_img_path, nonproject_obj_img_path):
        """
        Generates custom prompt payload for selecting an object from a list of objects

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_target_obj_mask_image = self.encode_image(nonproject_obj_img_path)
        base64_target_obj_bbox_image = self.encode_image(bbox_img_path)

        prompting_text_system = f"You are an expert in indoor design, and articulated furniture design.\n" + \
                                f"Your task is to count the number of doors (revolute) and drawers (prismatic) of an object/furniture." + \
                                f"I will give you an image showing a scene in our everyday life where an object in the same scene highlighted by a bounding box, and the mask of the object. " + \
                                f"Please tell me how many doors (those with revolute joint, that can rotate around an axis) and drawers (those with prismatic joint, that can slide along a direction) does the target object have.\n" + \
                                "When counting the number of doors and drawer, pay attention to the following:\n" + \
                                "1. Do not count closely positioned doors/drawers as one single doors/drawers:\n" + \
                                "E.g.(1). Do not regard several doors near each other as a single door. For example, two doors next to each other horizontally with opposite open direction should be defined as two doors, not one door.\n" + \
                                "E.g.(2). Do not regard several drawers stacked vertically or horizontally as a single drawer.\n" + \
                                "In other words, as long as a door or a drawer can be opened independently from other doors or drawers, it should be defined as a separate door or drawer.\n" + \
                                "2. A child link cannot be a door and a drawer at the same time. When you are not sure if a child link is a door or a drawer, choose the most likely one.\n" + \
                                "3. Please give the most appropriate answer without explaination.\n" + \
                                "Example output: (3 doors & 3 drawers)\n" + \
                                "Example output: (2 doors & 1 drawers)\n" + \
                                "Example output: (0 doors & 2 drawers)\n" + \
                                "Example output: (2 doors & 0 drawers)\n\n"

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        object_selection_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The following image shows the target object ({caption}) by a bounding box in a real world scene."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_target_obj_bbox_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"The following image shows the mask of the target object ({caption}), where only the target object is shown in its original color, while all other objects are masked as black. The mask might contain some noise.\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_target_obj_mask_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Based on the instructions I gave you, please tell me the door and drawer count for the target object in the correct format."
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return object_selection_payload

    def payload_nearest_neighbor(
            self,
            img_path,
            caption,
            bbox_img_path,
            candidates_fpaths,
            nonproject_obj_img_path,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)
        mask_obj_img = self.encode_image(nonproject_obj_img_path)

        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will present you an image showing a real-world scene, an object of interest in the scene, and a list of candidate orientations of an asset in a simulator.\n" + \
                        "Your task is to select the most geometrically similar asset as a digital twin of a real world object."

        prompt_user_1 = "### Task Overview ###\n" + \
                f"I will show you an image of a scene. " + \
                f"I will then present you an image of the same scene but with a target object ({caption}) highlighted by a bounding box, " + \
                "and another image showing the mask of the same object.\n" + \
                f"I will then present you a list of candidate assets in my simulator.\n" + \
                f"Your task is to choose the asset with highest geometric similarity to the target object ({caption}) so that I can use the asset to represent the target object in my simulator. " + \
                f"In other words, I want you to choose a digital twin for the target object ({caption}).\n\n" + \
                "### Special Requirements ###\n" + \
                "1. I have full control over these assets (as a whole), which means I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. " + \
                "For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio.\n" + \
                "2. When the target object is partially occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                "3. Also notice that the candidate asset snapshots are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                "4. Consider which asset, after being modified (reoriented, repositioned, rescaled, ratio changed, texture altered, relit), resembles the target object most closely. " + \
                "Geometry (shape) similarity after the aforementioned modifications is much more critical than appearance similarity.\n" + \
                "5. You should consider not only the overall shape, but also key features and affordance of the target object's category. " + \
                "For example, if it is a mug, consider if it has a handle and if some candidate assets have a handle. " + \
                "If they both have handles, which asset has the most similar handle as the target object.\n" + \
                "6. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                "Please return only the index of the most suitable asset's snapshot. Do not include explanations.\n" + \
                "Example output:2\n" + \
                "Example output:6\n" + \
                "Example output:1\n\n\n" + \
                "Now, let's take a deep breath and begin!\n"

        prompt_text_user_final = f"The following are a list of assets you can choose to represent the {caption}. " + \
                        f"Please choose the asset with highest geometric similarity to the target object ({caption}), i.e., choosing the best digital twin for the target object."

        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}"
                }
            },
            {
                "type": "text",
                "text": "The above image shows a scene in the real world. " + \
                        f"The following image shows the same scene with the target object ({caption}) highlighted by a bounding box."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the object is shown in its original color, and black pixels are other objects or background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{mask_obj_img}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]
        
        for i, candidate_fpath in enumerate(candidates_fpaths):
            text_prompt = f"image {i + 1}:\n"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            cand_base64 = self.encode_image(candidate_fpath)
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_base64}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload

    def payload_articulated_nearest_neighbor(
            self,
            img_path,
            caption,
            bbox_img_path,
            candidates_fpaths
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)

        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will present you an image showing a real-world scene, an object of interest in the scene, and a list of candidate orientations of an asset in a simulator.\n" + \
                        "Your task is to select the most geometrically similar asset as a digital twin of a real world object."

        prompt_text_user_1 = "### Task Overview ###\n" + \
                        f"I will show you an image showing a scene in the real world or another simulator, and a {caption} bounded by a bounding box in the scene. " + \
                        f"I will then present you a list of candidate assets in my simulator similar to the {caption}.\n" + \
                        f"Your task is to choose the asset with highest geometric similarity to the target object ({caption}) so that I can use the asset to represent the target object in my simulator. " + \
                        f"In other words, I want you to choose a digital twin for the target object ({caption}).\n\n" + \
                        "### Special Requirements ###\n" + \
                        "I have full control over these assets, which means that I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                        "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio. " + \
                        "When the target object is partly occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                        "Additionally, I cannot split a door or a drawer into two, or merge two doors or drawers into one. Nor can I transform a door into a drawer or vice versa. " + \
                        "Also notice that the assets are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                        f"The {caption} is an articulated object, meaning that it has doors, or drawers, or both. " + \
                        f"Therefore, when selecting the best asset, pay close attention to the following criteria: \n" + \
                        "1. Which asset has similar doors/drawers layout as the target object.\n" + \
                        "2. Handle type of each door/drawer.\n" + \
                        "3. After modification (reorientation, repositioning, rescaling, ratio change, texture alteration, relighting), which asset has the most similar (ideally identical) arrangement of drawers and doors as the target object, in terms of relative size and location. " + \
                        "Geometry similarity after the aforementioned modifications is much more critical than appearance similarity. \n" + \
                        "4. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                        "5. Please return only the index of the most suitable asset's snapshot. Do not include any explanation.\n" + \
                        "Example output:4\n\n\n" + \
                        "Now, let's take a deep breath and begin!\n"
        
        prompt_text_user_2 = "The above image shows a scene in the real world. " + \
                        f"The following image shows the same scene with the target object ({caption}) highlighted by a bounding box."
                           

        prompt_text_user_3 = f"The following are a list of assets you can choose to represent the {caption}." + \
                        f"Please choose the asset with highest geometric similarity to the target object ({caption}), i.e., choosing the best digital twin for the target object. Do not include explanation."

        content = [
            {
                "type": "text",
                "text": prompt_text_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_2
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_3
            }
        ]

        for i, candidate_fpath in enumerate(candidates_fpaths):
            text_prompt = f"image {i + 1}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            cand_base64 = self.encode_image(candidate_fpath)
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_base64}",
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload
    
    def payload_nearest_neighbor_pose(
            self,
            img_path,
            caption,
            bbox_img_path,
            nonproject_obj_img_path,
            candidates_fpaths,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor in terms of
        orientation.

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image
            candidates_fpaths (list of str): List of absolute paths to candidate images

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        bbox_base64 = self.encode_image(bbox_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        mask_obj_img = self.encode_image(nonproject_obj_img_path)

        prompt_text_system = "You are expert in orientation estimation and feature matching.\n\n" + \
                    "### Task Description ###\n" + \
                    f"The user will present you an image showing a real-world scene. " + \
                    "The user will then show an image of the same scene with an object highlighted by a bounding box, and the mask of the same object. " + \
                    "This object is represented in a simulator by a digital twin asset. " + \
                    "The user can re-oriente the asset (rotate the asset around its local z axis), and rescale it to accurately match the target object in the input image.\n\n" + \
                    "The user aims to match the orientation of the asset to the target object in the camera frame (i.e., from the viewer's ponit of view). " + \
                    "The user will present you a list of candidate orientations.\n\n" + \
                    "Your task is to select the best orientation of the asset from the candidate orientations that best match the orientation of the real target object from the viewer's point of view.\n\n" + \
                    "### Special Considerations ###\n" + \
                    "Please keep the following in mind:\n\n" + \
                    "1. There might be other objects in the image. Please focus on the target object.\n\n" + \
                    "2. Please select the best orientation in the camera frame, i.e., the orientations are with respect to the viewer. " + \
                    "For example, if the image is taken from a 45 degree lateral view from the left of the target object's frontal face, " + \
                    "then you should select the orientation where the asset is also observed from a 45 degree lateral view from left of the asset's frontal face.\n" + \
                    "If the target object is angled to the left, then you should select the orientation that the asset is also angled to the left viewing from the camera.\n" + \
                    "If the target object is angled to the right, then you should select the orientation that the asset is also angled to the right viewing from the camera.\n" + \
                    "If the object in the input image faces the camera (does not angled to left or right), then you should select the orientation where the asset faces the camera.\n" + \
                    "So on and so forth.\n\n" + \
                    "3. The candiates may not have a perfect orientation. Please select the nearest one.\n" + \
                    "For example, if the image is taken from a 45 degree lateral view from the left of the target object's frontal face, " + \
                    "but there may be no candidate snapshot taken from a 45 degree lateral view from the left of the corresponding asset. " + \
                    "Suppose there are only frontal view snapshot and snapshot taken from a 45 degree lateral view from the right of the corresponding asset, " + \
                    "please select the frontal view because it has smaller orientation difference with the correct orientation (front to left is smaller than right to left).\n\n" + \
                    "4. When selecting the best orientation, you should first identify common features of the digital twin asset and the target object that can define 'orientation'. " + \
                    "For example, cabinets have the same orientation if their frontal faces are facing the same direction (from the viewer's point of view), where the face with doors and drawers are usually considered as the frontal face of a cabinet. " + \
                    "For other objects, you should also consider key features that define orientation of the category, like the back of a chair, and the handle of a spoon.\n\n" + \
                    "5. Parts of the object may be occluded. Please use partially observable features or common sense to identify key features for determining orientation. " + \
                    "For instance, if the handles of a wardrobe are occluded, you can infer the frontal face by 'the face with the doors/drawers is usually the frontal face'. " + \
                    "If even the doors are occluded, apply common sense, such as 'the back of a wardrobe usually faces the wall, so the opposite face is likely the front,' to determine the frontal face and orientation.\n\n" + \
                    "6. You only need to consider orientation, not rescaling. Keep in mind that the user can rescale the asset along each directions without keeping the relative ratio after you determine the orientation. " + \
                    "For example, a sofa asset may be wide, while the real world sofa may be narrow. " + \
                    "After you determine the orientation, the user can rescale the sofa asset along its local horizontal axis to make it as narrow as the real world sofa. " + \
                    "Thus when you select the best orientation, you should focus on their common features (i.e., the back if they both have a back) even though one of them is wider.\n\n" + \
                    "7. Please return only the index of the most suitable orientation of the digital twin without explanations. The index of candidate orientations start from 0.\n\n" + \
                    "Example output:4"

        content_system = [
            {
                "type": "text",
                "text": prompt_text_system
            }
        ]

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The above image shows a scene in the real world. The following image shows the target object ({caption}) bounded by a bounding box."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the object is shown in its original color, while all other objects and background are masked out as black."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{mask_obj_img}"
                }
            },
            {
                "type": "text",
                "text": "The following images show candidate orientations of the digital twin asset with starting index 0:\n\n"
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"orientation {i}:\n"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
                "type": "text",
                "text": "Please take a deep breath, and now please select the nearest orientation that best matches the target object from the viewer's point of view without explanation. Please strictly follow all instructions.\n"
            })

        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return NN_payload
    
    def payload_filter_wall(
            self,
            img_path,
            candidate_fpath,
    ):
        """
        Prompt determining whether a mask is part of a wall/backsplash

        Args:
            img_path (str): Absolute path to image to infer object selection from
            candidate_fpath (str): Absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        original_img_base64 = self.encode_image(img_path)
        cand_img_base64 = self.encode_image(candidate_fpath)

        prompt_text_system = "You are an expert in indoor design and plane fitting.\n\n" + \
                    "The user will provide an image and a mask. Your task is to determine if the mask is a wall or backsplash in the original image."
        
        prompt_text_user = "### Task Description ###\n" + \
                        "I will provide an image showing a scene, and a mask where only the target wall or backsplash is shown in its original color, while all other objects are masked as black.\n" + \
                        "Your task is to distinguish if the mask is part of a wall or backsplash. If yes, return y; If no, return n.\n\n" + \
                        "### Special Requirements ###\n" + \
                        "Pay attention to the following:\n" + \
                        "1. Pixels with their original color are pixels belonging to the mask, while black pixles are outside the mask.\n" + \
                        "2. It is fine if the mask contains pixels belonging to another wall or other objects, but if more than half of the mask contains pixels of other objects, please return n.\n" + \
                        "3. The mask does not need to include a whole wall/backsplash. As long as it includes a part of a wall/backsplash without including a significant part of other objects, please return y.\n\n" + \
                        "### Example Outputs ###\n" + \
                        "Example output (If the mask is part of a wall/backsplash): y\n" + \
                        "Example output (If the mask is not part of a wall/backsplash): n\n\n\n" + \
                        "Now take a deep breath and begin!"

        content_system = [{
            "type": "text",
            "text": prompt_text_system
        }]

        content = [
            {
                "type": "text",
                "text": prompt_text_user
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{original_img_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The above image shows a scene in the real world.\n" + \
                    "The following image shows a possible mask for a wall/backsplash in the image. Please determine if it is part of a wall/backsplash following given instructions."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_img_base64}",
                }
            }
        ]

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }
        return payload
    
    def payload_mount_type(
            self,
            caption,
            bbox_img_path,
            obj_and_wall_mask_path,
            candidates_fpaths
    ):
        """
        Prompt determining whether an object is on the floor or mounted on the wall (and which wall)

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            obj_and_wall_mask_path (str): Absolute path to segmented object and walls image
            candidates_fpaths (list of str): List of absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        bbox_base64 = self.encode_image(bbox_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        obj_and_wall_mask_base64 = self.encode_image(obj_and_wall_mask_path)

        # Surprisingly, putting instructions under system role works better
        prompting_text_system = "You are an expert in indoor design, orientation estimation, and plane fitting.\n\n" + \
                        "### Task Description ###\n" + \
                        "The user will present an image showing a scene from the real world, with an object highlighted by a bounding box. " + \
                        "The user wants to reconstruct this scene in a simulator, so it's crucial to determine whether the object is installed/fixed on wall/backsplash or not.\n" + \
                        "An object is either installed/fixed on wall (e.g., a picture hung on a wall, a kitchen cabinet installed on a backsplash, a cabinet whose back is installed on a wall and side is installed on another wall), or not (e.g., a table put on the floor, a coffee machine on a table).\n" + \
                        "The user will install all objects installed/fixed on wall and disable it from falling down when importing in the simulator; " + \
                        "for objects not installed/fixed on wall, the user will put the object on object/floor beneath it.\n" + \
                        "The user will sequentially show you masks of each wall/backsplash plane in the scene, where a proportion of each wall/backsplash is shown in its original color, and all other elements are masked in black.\n" + \
                        "Note that only a proportion of each wall is included in the corresponding mask. You should regard each wall/backsplash as a whole plane, not only the proportion covered by the mask.\n" + \
                        "Your task is to classify the target object based on the following two options:\n\n" + \
                        "Type 1: If the target object is installed/fixed on a wall/backsplash's plane, return 'wall' followed by the index of the plane(s) the object is installed/fixed on.\n\n" + \
                        "If the object is installed/fixed on multiple walls, return the indices of the walls separated by a comma, e.g., 'wall1,wall3'. " + \
                        "This happens when an object is installed at the corner of two orthogonal wall planes, where the back face of the object is installed on a wall, and a side face is installed on the orthogonal wall.\n\n" + \
                        "Type 2: If the target object is not installed/fixed on the wall/backsplash's plane, meaning it rests on the floor or other objects, such that the object would fall if everything below it were removed, return 'floor'. " + \
                        "Be cautious: an object close to the floor might still be installed/fixed on a wall/backsplash.\n\n" + \
                        "### Special Requirements ###\n" + \
                        "Please keep the following in mind:\n" + \
                        "1. You can use common sense when determining the type. For instance, fridges are rarely installed/fixed on a wall, but usually put on the floor (Type 2) although possibly aligned with a wall, whereas cabinets could fit any of the two types; " + \
                        "Objects placed on top of other objects are rarely installed/fixed on a wall, like objects on tables and countertops.\n" + \
                        "These are heuristics from common sense, but if your observation clearly counter the above statements, please follow the observation.\n" + \
                        "2. For an object with doors and drawers like cabinets, wardrobes and refrigerators, pay attention to see if its back is installed on a wall/backsplash plane. " + \
                        "Sometimes the back of such an object is installed on the plane, but due to occlusion you cannot see where the object masks physical contact with the plane.\n" + \
                        "3. Treat each wall/backsplash as an entire 3D plane. The wall mask may only show part of the plane, and the area an object makes physical contact with a wall (Type 1) may not be visible. For instance, in a kitchen, a backsplash may only be a small part of a larger plane that multiple objects (e.g., cabinets) are installed/fixed on. " + \
                        "Such objects should be classified as Type 1 even if the contact area is not included in the mask, or even far away from the mask.\n" + \
                        "4. You can refer to objects belonging to the same category of the target object and placed closely to the target object in the scene as reference. " + \
                        "Multiple instances of the same object category placed together usually has the same installation type. " + \
                        "For example, multiple cabinets aligned horizontally are either all Type 1 or all Type 2. " + \
                        "A typical example would be all top and bottom cabinets in a kitchen installed/fixed on the same backsplash plane.\n\n" + \
                        "5. Please provide the most appropriate answer without explanation.\n\n" + \
                        "### Example Outputs ###\n" + \
                        "Here are some example outputs corresponding to different mounting types:\n" + \
                        "Example output of Type 1 (Installed/fixed on a single wall): wall1\n" + \
                        "Example output of Type 1 (Installed/fixed on more than one walls): wall2,wall3\n" + \
                        "Example output of Type 2 (Not installed/fixed on a wall): floor"

        content_system = [{
            "type": "text",
            "text": prompting_text_system
        }]
        
        content = [
            {
                "type": "text",
                "text": f"The following image shows the target object ({caption}) bounded by a bounding box in a real world scene."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": "The following images show wall(s) in the image. Please regard walls as planes as mentioned in my instructions."
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"wall{i}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
            "type": "text",
            "text": f"To help you better decide if the target object ({caption}) is installed/fixed on one or multiple wall(s)/backsplash(es), " + \
                    "I also provide the following image shows all wall(s)/backsplash(es) and the target object in their original color, " + \
                    "while all other objects and planes are masked as black."
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{obj_and_wall_mask_base64}",
            }
        })

        content.append({
            "type": "text",
            "text": f"Now please choose the installation type of the target object ({caption}) following instructions."
        })

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return payload

    def payload_align_wall(
            self,
            caption,
            bbox_img_path,
            nonproject_obj_img_path,
            obj_and_wall_mask_path,
            candidates_fpaths
    ):
        """
        Prompt determining whether an object not fixed/installed on a wall is aligned with a wall (and which wall) to help adjust orientation

        Args:
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image
            obj_and_wall_mask_path (str): Absolute path to segmented object and walls image
            candidates_fpaths (list of str): List of absolute paths to wall masks

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        bbox_base64 = self.encode_image(bbox_img_path)
        obj_mask_base64 = self.encode_image(nonproject_obj_img_path)
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]
        obj_and_wall_mask_base64 = self.encode_image(obj_and_wall_mask_path)

        # Surprisingly, putting instructions under system role works better
        prompting_text_system = "You are an expert in indoor design, orientation estimation, distance estimation, and plane fitting.\n" + \
                "Your task is to determine if an object in a scene is both aligned with and makes physical contact with one or multiple walls or backsplashes in the scene." + \
                "### Task Description ###\n" + \
                "I will provide a real world image where an object is highlighted by a bounding box, and the caption and mask of the same object. " + \
                "I will then present masks of each wall or backsplash plane in the scene. In these masks, a portion of each wall or backsplash will be shown in its original color, while all other elements are shown in black. " + \
                "It's important to consider each wall/backsplash as an entire 3D plane, not just the portion shown in the mask.\n\n" + \
                "Your task is to determine if the target object highlighted by the bounding box is both aligned with and strictly touches one or multiple walls/backsplashes I shown you:\n\n" + \
                "Case 1: If the target object is both aligned with and touches a wall/backsplash plane, return 'wall' followed by the index of the plane the object makes physical contact with.\n\n" + \
                "If the object is aligned with and makes physical contact with multiple walls, return the indices of the walls separated by a comma, e.g., 'wall1,wall3'. " + \
                "This happens when an object is at the corner of two wall planes, where one face of the object is aligned with and makes physical contact with a wall, and another face is aligned with and makes physical contact with the adjacent wall.\n\n" + \
                "Case 2: Otherwise, return 'floor'.\n" + \
                "In other words, if an object is positioned along a wall(s)/backsplash(es) but does not make physical contact with it, you should return 'floor';\n" + \
                "Or if an object makes physical contact with a wall(s)/backsplash(es), but does not align with it (e.g., only a corner contacts a wall/backsplash), you should return 'floor'.\n\n" + \
                "### Technical Interpretation ###\n" + \
                "'Align with' might be vague in semantics. I will provide other interpretations here:\n" + \
                "'An object is aligned with a wall' means the normal vector of one of the faces of the object is the same as the wall's normal vector.\n" + \
                "Another interpretation would be: If an object is aligned with a wall, then the wall's normal vector must be on the object's local x or y axis.\n\n" + \
                "Also note that I am asking for wall(s)/bachsplash(es) that the target object is both aligned with and make a physical contact with. " + \
                "You should also make sure that the target object makes physical contact with all wall wall(s)/backsplash(es) you returned.\n\n" + \
                "### Special Considerations ###\n" + \
                "Please keep the following in mind:\n\n" + \
                "1. Consider a wall/backsplash as a single but entire 3D plane.\n" + \
                "You should regard a wall/backsplash as a single 3D plane, because there might be multiple 3D planes in the scene. Some of them may be other walls, while some of them might be partitions or screens. " + \
                "Other 3D planes may even be parallel with the wall/backsplash plane. You should only focus on the wall/backsplash covered by the mask.\n" + \
                "You should regard a wall/backsplash as an entire 3D plane because possibly only a portion of a wall/backsplash is visible in the mask. The area where an object makes a physical contact with a wall may not always be visible in the image. " + \
                "For instance, in a kitchen or other dining-related scenes, a backsplash might be a small part of a larger plane that includes multiple objects like cabinets and fridges. These objects may fall into Case 1, even if their contact area with the wall is not directly visible.\n\n" + \
                "2. It is possible that an object is aligned a wall plane but does not make a physical contact with it. Then you should not return that wall. " + \
                "Similarly, it is possible that an object is aligned with multiple wall planes, but only makes physical contact with one or two of them. Then you should only return walls that the object makes physical contact with.\n\n" + \
                "3. Here are useful criteria to determine if an object makes physical contact with a wall:\n" + \
                "(1) If there are other object(s) between the target object and the wall, then the object is impossible to make a physical contact with the wall, so you should not return that wall;\n" + \
                "(2) If you can see or infer the target object has distance with the wall, the object does not touch the wall, so you should not return that wall.\n" + \
                "4. Sometimes due to occlusion, the back of the target object may not be visible. " + \
                "When there is a wall/backsplash behind the target object, you should make inference based on the image and common sense if the back face makes a physical contact with the wall/backsplash behind it. " + \
                "This is common for objects with doors and drawers, like wardrobes, refrigerators, and cabinets. If they align with and make physical contact with one or multiple walls, " + \
                "you usually cannot see if they do touch the wall behind it. In this case, if you infer that the object is highly likely to make a physical contact with the wall behind it, or the distance is very small, you can return that wall.\n\n" + \
                "5. Pay attention to relatively large furnitures, like sofa, refrigerators, cabinets, wardrobes, and so on. They are more offen aligned with and make physical contact with wall(s)/backsplash(es). " + \
                "But this is my own experience. Please put your observation as higher priority.\n\n" + \
                "6. Only objects with a clear concept of 'faces' (e.g., 'frontal face', 'back') and the overall shape is a cuboid can be Case 1. For example, cabinets, fridges, and microwave ovens have a clear concept of faces, where the face with doors and drawers are usually considered as the frontal face. " + \
                "And their overall shape is roughly a cuboid. " + \
                "Objects without a clear definition of faces and objects whose overall shape is not a cuboid, such as bowls, cups, or flowers, should not be classified as aligned with and touching wall plane(s) (Case 1), as resizing them to fit a wall is impractical and unrealistic. (Imagine resizing a cup to fit the wall behind it could result in an unsymmetric cup that is long in the direction to the wall). " + \
                "Notice that I am not saying objects with 'faces' must be Case 1. For those objects, you should see if it is aligned with and makes a physical contact with the wall.\n\n" + \
                "7. For small moveble objects like cups, apples, and bags of chips, you should be very careful if you want to classify them as Case 1, because resizing them along the direction pointing toward the wall could lead to larger size in that direction. Since they are small in size, " + \
                "large changes of size in one direction could lead to unrealistic shape.\n\n" + \
                "8. The index of wall/backsplash planes will start from 0. Please provide the most appropriate answer without explanation.\n\n" + \
                "### Example Outputs ###\n" + \
                "Example output: wall1\n" + \
                "Example output: wall1,wall3\n" + \
                "Example output: floor"

        content_system = [{
            "type": "text",
            "text": prompting_text_system
        }]

        content = [
            {
                "type": "text",
                "text": f"The following image shows the target object ({caption}) bounded by a bounding box in a real world scene. Please focus on the target object ({caption})."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{bbox_base64}",
                }
            },
            {
                "type": "text",
                "text": f"The following image shows the mask of the target object ({caption}), where only the target object is shown in its original color, and black pixels are other objects or background."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{obj_mask_base64}",
                }
            },
            {
                "type": "text",
                "text": "The following images show all wall(s) in the image. Please regard walls as planes as mentioned in my instructions."
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"wall{i}:"
            text_dict = {
                "type": "text",
                "text": text_prompt
            }
            img_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{cand_imgs_base64[i]}"
                }
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append({
            "type": "text",
            "text": f"To help you better decide if the target object ({caption}) is aligned with and makes a physical contact with one or multiple wall(s)/backsplash(es), " + \
                    "I also provide the following image shows all wall(s)/backsplash(es) and the target object in their original color, " + \
                    "while all other objects and planes are masked as black."
        })

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{obj_and_wall_mask_base64}",
            }
        })

        content.append({
            "type": "text",
            "text": f"Now please respond following instructions."
        })

        payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 50
        }
        return payload
    
    @classmethod
    def extract_captions(cls, gpt_text):
        """
        Extracts captions from GPT text. Assumes text is a list.
        During prompting, we prompt GPT to give object captions in a list, so we can localize objects by localizing '[', ']' and ','

        Args:
            gpt_text (str): Raw GPT response, which assumes captions are included as part of a list

        Returns:
            list of str: Extracted captions
        """
        # Remove leading and trailing whitespaces and the surrounding brackets
        cleaned_str = gpt_text.strip().strip('[]')

        # Split the string by comma and strip extra spaces from each object
        raw_objects_list = [obj.strip() for obj in cleaned_str.split(',')]

        # Remove redundant quotes
        cleaned_strings = [str.strip("'").strip('"').lower() for str in raw_objects_list]
        objects_list = [f"{obj_name}" for obj_name in cleaned_strings]  # Enclose each string in double quotes

        return objects_list


    def payload_task_proposals(
            self,
            annotated_img_path,
            list_objects
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        annotated_scene_img_base64 = self.encode_image(annotated_img_path)
        

        prompt_text_system =  "You are an expert in task generation for the robot to learn in simulator. " + \
                              "The user will provide you with a list of objects present in a given scene and Scene Image.\n" + \
                              "Your task is to imagine a set of realistic, context-aware tasks that a robotic agent could perform in this scene, " + \
                              "The Tasks you generate should be reasonable in the scene."

        prompt_user_1 = "### Task Generation Overview Instructions ###\n" + \
                        "I will show you an image of a real-world Scene with annotated objects\n" + \
                        f"In this scene, have an Object list : {list_objects}.\n" + \
                        "Your goal is to infer the nature of the space (e.g., kitchen, workshop, office), and generate a diverse set of realistic tasks that could occur in that environment. \n" + \
                        "Each task should involve manipulation or interaction with one or more of the listed objects, or with logically related objects that fit naturally into the scene context.\n" + \
                        "Focus on manipulation or interaction with the object itself. Sometimes the object will have functions, e.g., a microwave can be used to heat food."+\
                        "in these cases, feel free to include other objects that are needed for the task. \n\n"   + \
                        "For each task you imagined, please write in the following format: \n" + \
                        "Task name: the name of the task. \n" + \
                        "Description: some basic descriptions of the tasks. \n" +\
                        "Additional Objects: Additional objects other than the provided Input Scene Objects. If no extra objects are needed, write 'None'.\n\n" + \
                        "### Special Requirements ###\n" + \
                        "1. You can assume that any object, including those not listed in the scene, can be provided as assets in the simulator. So feel free to include additional objects that are logically relevant and contextually appropriate for the task, even if they are not initially present in the scene.\n" +\
                        "2. Tasks can be complex or multi-step as long as they are feasible given the provided scene and object list.\n" + \
                        "3. All tasks must be contextually grounded in the scene. That is, they must make logical sense within the space and with the objects shown in the image.\n" + \
                        "4. Do not propose tasks that involve assembling or disassembling objects, repairing or checking functionality, or cleaning/maintaining the object. These are explicitly excluded.\n" + \
                        "5. Your focus should be on object manipulation or interaction — tasks where the robot actively uses, moves, adjusts, or places objects based on their affordance.\n" + \
                        "6. You are encouraged to be creative and propose high-standard tasks that go beyond generic or trivial actions. Avoid redundant or overlapping task ideas.\n" + \
                        "7. If a task involves logical or scene-fitting additional objects that are not in the provided object list, you may include them — but only if they *make sense* in the current context.\n\n" + \
                        "### Example Outputs ###\n" + \
                         "\n" + \
                        "## First Example Scenario ##\n" + \
                        "## First Scene ##\n" + \
                        "Input Scene Object : Cabinet_0, Cabinet_1,  Refrigerator, Microwave, cup, plate\n" + \
                        "\n" + \
                        "Task: Place a cup inside the microwave\n" + \
                        "Description: The robotic arm opens the microwave, places the cup inside, and closes the door.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Open refrigerator door\n" + \
                        "Description: The robotic arm will open the refrigerator door to access the items inside.\n" + \
                        "Additional Objects: None.\n" + \
                        "\n" + \
                        "Task: Store a cup in Cabinet\n" + \
                        "Description: The robotic arm picks up a cup and places it inside Cabinet_1 for storage.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task : Heat a hamburger Inside Oven \n" + \
                        "Description: The robot arm places a hamburger inside the oven, and sets the oven temperature to be appropriate for heating the hamburger.\n" + \
                        "Additional Objects: hamburger, oven\n" + \
                        "\n" + \
                        "Task: Heat food in microwave\n" + \
                        "Description: The robotic arm will place food inside the microwave and set the timer to heat it.\n" + \
                        "Additional Objects: food\n" + \
                        "\n" + \
                        "Task: Set the table with cup and plate\n" + \
                        "Description: The robotic arm places a cup and plate on the dining surface for a meal setting.\n" + \
                        "Additional Objects: dining table\n" + \
                        "#################################################################################\n" + \
                        "## Second Example Scenario ##\n" + \
                        "Input Example : \n" + \
                        "In Scene: Sofa, Coffee Table, Television, Rug, Picture Frame, Window, Curtain, Chair\n" + \
                        "\n" + \
                        "Output Example : \n" + \
                        "Task: Turn On TV\n" + \
                        "Description: The robotic arm will turn off the television after a movie or show is finished.\n" + \
                        "Additional Objects: television remote\n" + \
                        "\n" + \
                        "Task: Fold rug\n" + \
                        "Description: The robotic arm will fold the rug to either store it or clean the floor underneath.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Set up chair for sitting\n" + \
                        "Description: The robotic arm will adjust the chair to make it more comfortable for sitting, possibly by aligning it with the table.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Clean the coffee table\n" + \
                        "Description: The robotic arm uses a cloth to wipe the surface of the coffee table, removing dust or crumbs.\n" + \
                        "Additional Objects: cleaning cloth\n" + \
                        "\n" + \
                        "Task: Place a cup on the coffee table\n" + \
                        "Description: The robot arm picks up a cup and places it gently on the coffee table.\n" + \
                        "Additional Objects: cup\n" + \
                        "#################################################################################\n" + \
                        "## Third Example Scenario ##\n" + \
                        "Input Example:\n" + \
                        "In Scene: Desk, Computer, Chair, Printer, Books, Notebooks, Pen, Paper, Phone\n" + \
                        "\n" + \
                        "Output Example : \n" + \
                        "Task: Turn on the computer\n" + \
                        "Description: The robotic arm presses the power button on the computer tower or monitor to start it up.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Organize books on the desk\n" + \
                        "Description: The robotic arm arranges the books neatly into a vertical stack or shelf.\n" + \
                        "Additional Objects: bookshelf\n" + \
                        "\n" + \
                        "Task: Write notes in a notebook\n" + \
                        "Description: The robotic arm picks up a pen and writes on a notebook following a given text or instruction.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Turn on desk lamp\n" + \
                        "Description: The robotic arm flips the switch on a desk lamp to illuminate the workspace.\n" + \
                        "Additional Objects: desk lamp\n" + \
                        "\n" + \
                        "Task: Charge the phone\n" + \
                        "Description: The robotic arm connects the phone to a charging cable to begin charging.\n" + \
                        "Additional Objects: charging cable\n" + \
                        "\n" + \
                        "Task: Place a cup of coffee on the desk\n" + \
                        "Description: The robotic arm places a hot cup of coffee next to the computer for the user.\n" + \
                        "Additional Objects: coffee cup, coffee machine\n" + \
                        "#################################################################################\n" + \
                        "## Fourth Example Scenario ##\n" + \
                        "Input Example:\n" + \
                        "In Scene: Dining Table, Plates, Forks, Knives, Glasses, Napkins, Salt Shaker, Pepper Shaker, Bowl, Wine Glass\n" + \
                        "\n" + \
                        "Output Example:\n" + \
                        "Task: Fold napkins\n" + \
                        "Description: The robotic arm folds napkins into decorative shapes and places them next to each plate.\n" + \
                        "Additional Objects: None\n" + \
                        "\n" + \
                        "Task: Place a bowl of salad on the table\n" + \
                        "Description: The robotic arm brings a bowl filled with salad and sets it at the center of the table.\n" + \
                        "Additional Objects: salad\n" + \
                        "\n" + \
                        "Task: Pour wine into wine glasses\n" + \
                        "Description: The robotic arm carefully pours wine from a bottle into the wine glasses.\n" + \
                        "Additional Objects: wine bottle\n" + \
                        "\n" + \
                        "Task: Light a candle on the dining table\n" + \
                        "Description: The robotic arm uses a lighter to light a decorative candle to enhance the table ambiance.\n" + \
                        "Additional Objects: candle, lighter\n" + \
                        "\n" + \
                        "Task: Serve bread in a basket\n" + \
                        "Description: The robotic arm places slices of bread into a basket and puts it on the table.\n" + \
                        "Additional Objects: bread, bread basket\n" + \
                        "\n" + \
                        "Task: Pour juice into glasses\n" + \
                        "Description: The robotic arm pours juice from a pitcher into the glasses for each guest.\n" + \
                        "Additional Objects: juice\n"

        prompt_text_user_final = f"The following is an scene with a list of objects: {list_objects}\n" + \
                                 "Please generate a diverse and realistic set of tasks that a robot could perform in this space.\n" + \
                                 "Each task should involve manipulation or interaction with one or more of the listed objects, or with logically relevant additional objects that would fit naturally into the scene.\n\n" + \
                                 "Remember the goal is to reason about what kinds of purposeful, context-aware actions are possible in this scene.\n" + \
                                 "Make sure each task:\n" + \
                                 "- Is contextually grounded and plausible for the scene\n" + \
                                 "- Focuses on manipulation or interaction, not inspection, cleaning, or repair\n" + \
                                 "- Avoids redundancy or trivial actions\n" + \
                                 "- Can include additional objects that are relevant, assuming they are available in the simulator\n\n" + \
                                 "Follow this exact format for each task:\n" + \
                                 "Task: <the name of the task>\n" + \
                                 "Description: <a short, clear explanation of what the robot is doing>\n" + \
                                 "Additional Objects: <additional objects needed for the task, or 'None'>\n\n" + \
                                 "Now generate high-quality tasks based on the input scene."
        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{annotated_scene_img_base64}"
                }
            },
            {
                "type": "text",
                "text": "The above image shows a scene in the object bounding box annotation real world image. "
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]
        
        # for i, candidate_fpath in enumerate(candidates_fpaths):
        #     text_prompt = f"image {i + 1}:\n"
        #     text_dict = {
        #         "type": "text",
        #         "text": text_prompt
        #     }
        #     cand_base64 = self.encode_image(candidate_fpath)
        #     img_dict = {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/png;base64,{cand_base64}"
        #         }
        #     }
        #     content.append(text_dict)
        #     content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0.3
        }
        return NN_payload
    

    def payload_task_object_extraction(
            self,
            scene_objects,
            goal_task
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string


        prompt_text_system = "You are a robot & simulation expert. "

        prompt_user = """Carefully analyze the given scene objects and task to extract the objects needed to simulate the task.
However, the objects to be extracted must be other than scene objects.
Your answer should be in JSON format. For example:
{{"target_objects": ["obj_0", "obj_1"]}}

I will give you some examples

### Example 1 ###
Example Input:
Scene objects: 'cup', 'microwave', 'refrigerator', 'cabinet'
Task: Give me the bottle in the refrigerator

Example Output:
{{"target_objects": ["bottle"]}}

### Example 2 ###
Example Input:
Scene objects: 'monitor', 'notebook', 'pen', 'cup', 'keyboard', 'mouse', 'toy', 'snack', 'storage unit'
Task: Put the wipes in front of the keyboard

Example Output:
{{"target_objects": ["wipes"]}}

### Example 3 ###
Example Input:
Scene objects: 'table', 'chair', 'coffee machine', 'box', 'vacuum cleaner'
Task: Get me the box on the chair.

Example Output:
{{"target_objects": ["box"]}}

### Example 4 ###
Example Input:
Scene objects: 'sign', 'glass door', 'wall'
Task: Open the cabinet in front of the wall

Example Output:
{{"target_objects": ["cabinet"]}}

### Example 5 ###
Example Input:
Scene objects: 'cup', 'microwave', 'refrigerator', 'cabinet'
Task: Give me a dish in the cabinet above the microwave

Example Output:
{{"target_objects": ["dish"]}}

### Example 6 ###
Example Input:
Scene objects: 'locker'
Task: Open the cabinet next to the locker and give me the cup inside

Exmaple Output:
{{"target_objects": ["cabinet", "cup"]}}

### Example 7 ###
Example Input:
Scene objects: 'cup', 'microwave', 'cabinet'
Task: Give me the water bottle next to the microwave

Exmaple Output:
{{"target_objects": ["water bottle"]}}


Now, based on the provided scene objects, please identify which objects are necessary to execute the task.
Scene objects: {}
Task: {}
"""

        prompt_user_filled = prompt_user.format(scene_objects, goal_task)
        content = [
            {
                "type": "text",
                "text": prompt_user_filled
            }
        ]


        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 50
        }
        return NN_payload
    
    def payload_task_object_spatial_reasoning(
            self,
            annotated_image_path,
            scene_objects,
            goal_task,
            objects_to_be_placed
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        annotated_img_base64 = self.encode_image(annotated_image_path)

        prompt_text_system = """You are a professional simulation designer. The user will provide you with an annotated Scene image, a list of scene objects, objects to be placed, and a task.

First, carefully analyze the given task.
Then, understand the spatial relationships between objects through the Scene image and scene objects.
Next, based on your task analysis, determine the spatial relationships between the objects to be placed and the scene objects.
Finally, place the objects that need to be positioned in their optimal locations.

Please provide your answer as a single sentence of reasoning followed by a JSON format response.
"""

        step2_text_prompt1 = """When there are objects to be placed, please tell me the appropriate position of the object for the given scene and task.
I'll provide scene information as an image. The image will have annotated scene objects.
The placement position should be based on a parent_object, and the parent_object should be chosen from either a scene object or one of the objects to be placed.
Position should be answered as one of ['above', 'below', 'left', 'right', 'front', 'back', 'inside'] relative to the parent_object.
Think through various perspectives, derive results for all possible scenarios, and explain your reasoning in one sentence.

Let me give you some examples

### Example 1 ###
Example Input:
Scene image: <image>
Scene objects: desk_0, drawer_0, notebook_0, notebook_1, monitor_0
Objects to be placed: ['pencil', 'pencil case']
Task: Take a pencil out of the pencil case on the desk.

Example Output:
After considering the task and spatial relationships, I need to place the pencil case on the desk and show the pencil being taken out from inside the pencil case.
{
	"scenario_0": {
		"objects": {
		    "pencil case": {
				"parent_object" : "desk_0",
				"placement" : "above"
		    },
			"pencil": {
				"parent_object" : "pencil case",
				"placement" : "inside"
			}
		}
	}
}

### Example 2 ###
Example Input:
Scene image: <image>
Scene objects: refrigerator_0, microwave_0, oven_0, cabinet_0, cabinet_1, cabinet_2, cabinet_3
Objects to be placed: ['dish']
Task: Give me a dish in the cabinet above the microwave

Example Output:
Based on the scene image analysis, cabinets above the microwave are cabinet_0 and cabinet_4.
{
	"scenario_0": {
		"objects": {
		    "dish": {
				"parent_object" : "cabinet_0",
				"placement" : "inside"
		    }
		}
	},
	"scenario_1": {
		"objects": {
		    "dish": {
				"parent_object" : "cabinet_4",
				"placement" : "inside"
		    }
		}
	}
}

### Example 3 ###
Example Input:
Scene image: <image>
Scene objects: table_0, bookshelf_0, bookshelf_1, bookshelf_2, bookshelf_3, bookshelf_4, bookshelf_5, computer_0, keyboard_0
Objects to be placed: ['book']
Task: Put the book on the bookshelf next to the computer.

Example Output:
After analyzing the task and spatial relationships in the scene, I can determine that the book should be placed on a bookshelf adjacent to the computer.
{
	"scenario_0": {
		"objects": {
		    "book": {
				"parent_object" : "bookshelf_0",
				"placement" : "inside"
		    }
		}
	},
	"scenario_1": {
		"objects": {
		    "book": {
				"parent_object" : "bookshelf_1",
				"placement" : "inside"
		    }
		}
	},
	"scenario_2": {
		"objects": {
		    "book": {
				"parent_object" : "bookshelf_4",
				"placement" : "inside"
		    }
		}
	},
	"scenario_0": {
		"objects": {
		    "book": {
				"parent_object" : "bookshelf_5",
				"placement" : "inside"
		    }
		}
	}
}

### Example 4 ###
Example Input:
Scene image: <image>
Scene objects: 'table_0', 'chair_0', 'coffee_machine_0', 'box_0', 'vacuum_cleaner_0'
Objects to be placed: ['coffee capsule', 'drawer']
Task: Take a coffee capsule out of the drawer next to the coffee machine.

Example Output:
After analyzing the task and spatial relationships, I need to place a drawer next to the coffee machine and position the coffee capsule inside the drawer to simulate taking it out.
{
	"scenario_0": {
		"objects": {
		    "drawer": {
				"parent_object" : "coffee_machine_0",
				"placement" : "left"
		    },
			"coffee capsule": {
				"parent_object" : "drawer",
				"placement" : "inside"
			}
		}
	},
	"scenario_1": {
		"objects": {
		    "drawer": {
				"parent_object" : "coffee_machine_0",
				"placement" : "right"
		    },
			"coffee capsule": {
				"parent_object" : "drawer",
				"placement" : "inside"
			}
		}
	}
}

### Example 5 ###
Example Input:
Scene image: <image>
Scene objects: 'microwave_0', 'table_0', 'table_1', 'cabinet_0', 'cabinet_1', 'cabinet_2', 'cabinet_3' 'oven_0', 'knife_0', 'refrigerator_0'
Objects to be placed: ['dish', 'food']
Task: Grab a food on the plate and put into the refrigerator.

Example Output:
After analyzing the task of grabbing food from a plate and transferring it to the refrigerator, the food should initially be positioned on the dish which is placed on a table.
{
	"scenario_0": {
		"objects": {
		    "dish": {
				"parent_object" : "table_0",
				"placement" : "above"
		    },
			"food": {
				"parent_object" : "dish",
				"placement" : "above"
			}
		}
	},
	"scenario_1": {
		"objects": {
		    "dish": {
				"parent_object" : "table_1",
				"placement" : "above"
		    },
			"food": {
				"parent_object" : "dish",
				"placement" : "above"
			}
		}
	},
}

### Example 6 ###
Example Input:
Scene image: <image>
Scene objects: 'desk_0', 'cabinet_0', 'monitor_0', 'computer_0', 'keyboard_0', 'notebook_0', 'notebook_1', 'tumbler_0'
Objects to be placed: ['cabinet', 'book']
Task: Take a book out of the cabinet and put it on the desk.

Example Output:
After analyzing the task of taking a book out of a cabinet and putting it on the desk, I need to place the cabinet in an accessible location near the desk and position the book inside the cabinet.
{
	"scenario_0": {
		"objects": {
		    "cabinet": {
				"parent_object" : "desk_0",
				"placement" : "below"
		    },
			"book": {
				"parent_object" : "cabinet_0",
				"placement" : "inside"
			}
		}
	}
}


Now look at the Scene image, Scene objects, Objects to be placed, and the Task to provide the correct answer.
Remember that parent_object can be used from not only scene objects but also from objects to be placed. And make sure the object placements are appropriate and well-aligned with the given task.
Scene image:
"""

        step2_text_prompt2 = f'Scene objects: {scene_objects}' +\
                            f'Objects to be placed: {objects_to_be_placed}' +\
                            f'Task: {goal_task}'
        
        content = [
            {
            "type": "text",
            "text": step2_text_prompt1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{annotated_img_base64}"
                }
            },
            {
            "type": "text",
            "text": step2_text_prompt2
            },
        ]

        text_dict_system = {
                "type": "text",
                "text": prompt_text_system
            }
        

        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 1000
        }
        return NN_payload
    
    def payload_task_object_extraction_and_spatial_reasoning(
            self,
            annotated_image_path,
            scene_objects,
            goal_task
    ):
        """
        

        Args:
            

        Returns:
            
        """
        # Getting the base64 string
        annotated_img_base64 = self.encode_image(annotated_image_path)

        prompt_text_system = """You are a professional simulation designer.
Look at the task and scene and think of new objects necessary to perform the task successfully.

If the objects already in the scene are sufficient for the task and the task is clearly needed in the scene, you can leave the "objects" value empty.
However, if the scene does not currently reflect a need for the task, you must introduce appropriate objects to create the initial conditions for that task.

Place the objects you've thought of in appropriate placement to perform the given task.
Placement must be selected strictly from only these seven 3D spatial options: ['above', 'below', 'left', 'right', 'front', 'back', 'inside'], relative to the parent_object.
These placements should be interpreted in 3D space from the viewer’s perspective, not based on the 2D layout of a surface. For example, an object placed on the top surface of a desk should be labeled as 'above', not 'left' or 'right', even if it's visually on the right side of the desk.
The parent_object can be either a scene object or a newly thought of object.

Please provide answers for all possible scenarios.
Your answer should consist of a reasoning explanation followed by the JSON. Keep explanation simple and don't add any additional text after the JSON.

Keep in mind that your job is to set up the initial state of the simulation environment. Therefore, you should describe the appropriate initial state to perform the task, not the state after the task is completed.
You are setting up the initial state of the environment where the task has not yet been performed. This means the scene should clearly reflect the need for the task.
"""

        step1_and_2_text_prompt1 = """
Let me give you some examples

### Example 1 ###
Example Input:
Task: Give me the water bottle next to the microwave
Scene image: <image>
Scene objects: ["cabinet_0", "cabinet_1", "cabinet_2", "cabinet_3", "cabinet_4", "microwave_0", "refrigerator_0", "cup_0"]

Example Output:
A water bottle should be placed to the right or left of the microwave on the countertop so that it's clearly next to it and easy to pick up.
{{
    "scenario_0": {{
        "objects": {{
            "water_bottle": {{
                "parent_object": "microwave_0",
                "placement": "right"
            }}
        }}
    }},
    "scenario_1": {{
        "objects": {{
            "water_bottle": {{
                "parent_object": "microwave_0",
                "placement": "left"
            }}
        }}
    }}
}}

### Example 2 ###
Example Input:
Task: Clear the table
Scene image: <image>
Scene objects: ["table_0", "chair_0", "vacuum_cleaner_0", "box_0", "coffee_machine_vacuum_0", "chair_1"]

Example Output:
In the current scene, the table has objects (box_0 and coffee_machine_vacuum_0) on it that need to be cleared. To perform this task, we need a destination for these objects. Since the scene lacks any storage or surface to move these items to, we must introduce appropriate objects (e.g., a "counter" or "cabinet") in the initial state, so that the task of clearing the table can be carried out.
{{
    "scenario_0": {{
        "objects": {{
            "cabinet": {{
                "parent_object": "table_0",
                "placement": "right"
            }}
        }}
    }},
    "scenario_1": {{
        "objects": {{
            "side_table": {{
                "parent_object": "chair_0",
                "placement": "left"
            }}
        }}
    }}
}}

### Example 3 ###
Example Input:
Task: Open the cabinet next to the locker and give me the cup inside
Scene image: <image>
Scene objects: ["locker_0"]

Example Output:
There is no visible cabinet next to the locker in the current scene, so to successfully perform the task, a cabinet containing a cup must be added adjacent to the locker.
{{
    "scenario_0": {{
        "objects": {{
            "cabinet": {{
                "parent_object": "locker_0",
                "placement": "left"
            }},
            "cup": {{
                "parent_object": "cabinet",
                "placement": "inside"
            }}
        }}
    }}
}}


Now based on the following input, please give me appropriate answer.
Task: {}
Scene image: """

        step1_and_2_text_prompt2 = """Scene objects: {}
"""

        # Fill text prompts
        step1_and_2_text_prompt1_filled = step1_and_2_text_prompt1.format(goal_task)
        step1_and_2_text_prompt2_filled = step1_and_2_text_prompt2.format(scene_objects)

        content = [
            {
            "type": "text",
            "text": step1_and_2_text_prompt1_filled
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{annotated_img_base64}"
                }
            },
            {
            "type": "text",
            "text": step1_and_2_text_prompt2_filled
            },
        ]

        text_dict_system = {
                "type": "text",
                "text": prompt_text_system
            }
        content_system = [text_dict_system]


        task_object_extraction_and_spatial_reasoning_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 500
        }
        return task_object_extraction_and_spatial_reasoning_payload
    
    def payload_task_object_resizing(
            self,
            object_size_info,
            goal_task
    ):
        """
        
        """

        prompt_text_system = """You are an expert in robotics, object modeling, and real-world dimensions. Your role is to determine appropriate sizes for objects used in robotic tasks.

The user will provide:
- A task description that explains what the robot is doing.
- A list of objects, each with a name and its size (longest dimension) in meters.
- User input will be the following format:

Task: [task description]
[object_1_name] ([current_size])
[object_2_name] ([current_size])
...

Your job is to:
1. Use real-world knowledge to identify the typical or appropriate size (longest dimension) for each object in the given task context.
2. Output your answer strictly in the following format:

[One-sentence reasoning about the task and object sizing]
object_1 (appropriate_size_in_meters)
object_2 (appropriate_size_in_meters)
...

Only respond using this format. Do not include any additional explanation or text outside the required structure.
"""

        prompt_user =   """Task: {}
{}
"""
        obj_info_str = ""
        for obj_name, obj_dims in object_size_info.items():
            obj_info_str_ = "{} ({:.4f})\n".format(obj_name, max(obj_dims))
            obj_info_str += (obj_info_str_)
        obj_info_str += ("\nWhat is the appropriate real-world size (longest dimension) of each object for this task?")

        prompt_user_filled = prompt_user.format(goal_task, obj_info_str)

        content = [
            {
                "type": "text",
                "text": prompt_user_filled
            }
        ]


        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 100
        }
        return NN_payload
    
    def payload_nearest_neighbor_text_ref_scene(
            self,
            sim_real_img_path,
            goal_task,
            parent_obj_name,
            placement,
            caption,
            candidates_path,
            top_k = 3
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        sim_real_scene_img_base64 = self.encode_image(sim_real_img_path)
        # parent_obj_bbox_img_base64 = self.encode_image(parent_obj_bbox_img_path)
        # f"I will provide an image with a bounding box highlighting the location of the parent object ({parent_obj_name}) where the target object will be placed."+ \
        candidate_obj_img_base64 = self.encode_image(candidates_path)


        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will provide you with an image showing real-world scene and simulation scene, and a list of candidate orientations of an asset in the simulator.\n" + \
                        f"Your task is to select the top {top_k} candidate assets that best match the goal, placement requirements, and overall context of the given scene."

        prompt_user_1 = "### Task Overview ###\n" + \
                f"I will show you an image of a real-world scene. " + \
                f"I will show you an image of a simulation scene. \n" + \
                f"In this scene, the goal task is to: {goal_task}. " + \
                f"To achieve this, the object needs to be placed {placement} the {parent_obj_name}. \n" + \
                f"I will then present you a list of candidate assets in my simulator. \n" + \
                f"Your task is to select the top {top_k} candidate assets that have the highest geometric similarity to the target object ({caption}), in descending order of similarity so that I can use the asset to represent the target object in the simulator with the intended goal and placement. " + \
                f"In other words, I want you to select the most suitable object, taking into account the scene, task, and placement.\n\n" + \
                "### Special Requirements ###\n" + \
                "1. I have full control over these assets (as a whole), which means I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. " + \
                "For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio.\n" + \
                "2. When the target object is partially occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                "3. Also notice that the candidate asset snapshots are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                "4. Consider which asset, after being modified (reoriented, repositioned, rescaled, ratio changed, texture altered, relit), resembles the target object most closely. " + \
                "Geometry (shape) similarity after the aforementioned modifications is much more critical than appearance similarity.\n" + \
                "5. You should consider not only the overall shape, but also key features and affordance of the target object's category. " + \
                "For example, if it is a mug, consider if it has a handle and if some candidate assets have a handle. " + \
                "If they both have handles, which asset has the most similar handle as the target object.\n" + \
                "6. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                "Please return exactly {top_k} indices of the most suitable asset snapshots, in descending order of similarity. Only include the indices, separated by commas. Do not include any explanations. \n" + \
                "Example output:2, 14, 21\n" + \
                "Example output:6, 31, 1\n" + \
                "Example output:16\n" + \
                "Example output:3, 5, 7\n" + \
                "Example output:10, 3, 4, 6, 17, 24\n" + \
                "Example output:1, 3, 19, 32\n\n\n" + \
                "Now, let's take a deep breath and begin!\n"

        prompt_text_user_final = f"The following are a list of assets you can choose to represent the {caption}. " + \
                        f"Please select the top {top_k} assets that best fits the scene, considering the intended task ({goal_task})," +\
                        f"the placement ({placement} of the {parent_obj_name}), and the geometric similarity to the target object.\n" +\
                        "Choose the most suitable object for this context."
        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{sim_real_scene_img_base64}"
                }
            },
            {
                "type": "text",
                "text": "The above image left side shows a scene in the real world. " + \
                        f"and right side shows the simulation scene that is similar to the real-world scene."
            },
            {
                "type": "text",
                "text": f"The following image shows candidate asset list ({caption})."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{candidate_obj_img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]
        
        # for i, candidate_fpath in enumerate(candidates_fpaths):
        #     text_prompt = f"image {i + 1}:\n"
        #     text_dict = {
        #         "type": "text",
        #         "text": text_prompt
        #     }
        #     cand_base64 = self.encode_image(candidate_fpath)
        #     img_dict = {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/png;base64,{cand_base64}"
        #         }
        #     }
        #     content.append(text_dict)
        #     content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload
    
    def payload_nearest_neighbor_text_ref_scene_bbox(
            self,
            sim_real_img_path,
            parent_obj_bbox_img_path,
            goal_task,
            parent_obj_name,
            placement,
            caption,
            candidates_path,
            top_k = 3
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        sim_real_scene_img_base64 = self.encode_image(sim_real_img_path)
        parent_obj_bbox_img_base64 = self.encode_image(parent_obj_bbox_img_path)
        
        candidate_obj_img_base64 = self.encode_image(candidates_path)


        prompt_text_system = "You are an expert in indoor design and feature matching. " + \
                        "The user will provide you with an image showing real-world scene and simulation scene, and a list of candidate orientations of an asset in the simulator.\n" + \
                        f"Your task is to select the top {top_k} candidate assets that best match the goal, placement requirements, and overall context of the given scene."

        prompt_user_1 = "### Task Overview ###\n" + \
                f"I will show you an image of a real-world scene. " + \
                f"I will show you an image of a simulation scene. \n" + \
                f"In this scene, the goal task is to: {goal_task}. " + \
                f"To achieve this, the object needs to be placed {placement} the {parent_obj_name}. \n" + \
                f"I will provide an image with a bounding box highlighting the location of the parent object ({parent_obj_name}) where the target object will be placed."+ \
                f"I will then present you a list of candidate assets in my simulator. \n" + \
                f"Your task is to select the top {top_k} candidate assets that have the highest geometric similarity to the target object ({caption}), in descending order of similarity so that I can use the asset to represent the target object in the simulator with the intended goal and placement. " + \
                f"In other words, I want you to select the most suitable object, taking into account the scene, task, and placement.\n\n" + \
                "### Special Requirements ###\n" + \
                "1. I have full control over these assets (as a whole), which means I can reoriente, reposition, and rescale the assets; I can also change the relative ratios of length, width, and height; adjust the texture; or relight the object by defining a new light direction; " + \
                "It's important to note that the aforementioned operations can only be applied to the entire object, not to its parts. " + \
                "For example, I can rescale an entire cabinet without keeping the original length-width-height ratio, but I cannot rescale one drawer of a cabinet by one ratio and another drawer by a different ratio.\n" + \
                "2. When the target object is partially occluded by other objects, please observe its visible parts and infer its full geometry.\n" + \
                "3. Also notice that the candidate asset snapshots are taken with a black background, so pay attention to observe the asset snapshot when it has a dark color.\n" + \
                "4. Consider which asset, after being modified (reoriented, repositioned, rescaled, ratio changed, texture altered, relit), resembles the target object most closely. " + \
                "Geometry (shape) similarity after the aforementioned modifications is much more critical than appearance similarity.\n" + \
                "5. You should consider not only the overall shape, but also key features and affordance of the target object's category. " + \
                "For example, if it is a mug, consider if it has a handle and if some candidate assets have a handle. " + \
                "If they both have handles, which asset has the most similar handle as the target object.\n" + \
                "6. Please ensure you return a valid index. For example, if there are n candidates, then your response should be an integer from 1 to n." + \
                "Please return exactly {top_k} indices of the most suitable asset snapshots, in descending order of similarity. Only include the indices, separated by commas. Do not include any explanations. \n" + \
                "Example output:2, 14, 21\n" + \
                "Example output:6, 31, 1\n" + \
                "Example output:16\n" + \
                "Example output:3, 5, 7\n" + \
                "Example output:10, 3, 4, 6, 17, 24\n" + \
                "Example output:1, 3, 19, 32\n\n\n" + \
                "Now, let's take a deep breath and begin!\n"

        prompt_text_user_final = f"The following are a list of assets you can choose to represent the {caption}. " + \
                        f"Please select the top {top_k} assets that best fits the scene, considering the intended task ({goal_task})," +\
                        f"the placement ({placement} of the {parent_obj_name}), and the geometric similarity to the target object.\n" +\
                        "Choose the most suitable object for this context."
        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{sim_real_scene_img_base64}"
                }
            },
            {
                "type": "text",
                "text": "The above image left side shows a scene in the real world. " + \
                        f"and right side shows the simulation scene that is similar to the real-world scene."
            },
            {
                "type": "text",
                "text": f"The following image shows the bounding box of the parent object({parent_obj_name})."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{parent_obj_bbox_img_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The following image shows candidate asset list ({caption})."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{candidate_obj_img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]
        
        # for i, candidate_fpath in enumerate(candidates_fpaths):
        #     text_prompt = f"image {i + 1}:\n"
        #     text_dict = {
        #         "type": "text",
        #         "text": text_prompt
        #     }
        #     cand_base64 = self.encode_image(candidate_fpath)
        #     img_dict = {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/png;base64,{cand_base64}"
        #         }
        #     }
        #     content.append(text_dict)
        #     content.append(img_dict)

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0,
            "max_tokens": 10
        }
        return NN_payload
    

    def payload_front_view_image(
            self,
            candidate_view_path,
            goal_task,
            parent_obj_name,
            placement,
            caption,
            direction="front",
    ):
        """
        Generates custom prompt payload for selecting an object from a list of objects

        Args:
            img_path (str): Absolute path to image to infer object selection from
            obj_list (list of str): List of previously detected objects
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        candidate_view_img_base64 = self.encode_image(candidate_view_path)

        prompting_text_system = "You are an expert in determining the orientation of objects.\n\n" + \
                            "### Task Overview ###\n" + \
                            "The user will show you four images of the object taken from different directions: front, left, right, and back, in a random order. " + \
                            "Each image you see contains four different views of the object (front, left, right, and back) combined into a single image. " + \
                            "Each view is labeled with a number from 1 to 4. \n" + \
                            "Your task is to select the image that best represents the {direction} view of the target object. \n\n" + \
                            "### Special Requirements ###\n\n" + \
                            "Please follow these guidelines when selecting your answer:\n\n" + \
                            "1. Select the image that best represents the {direction} view of the object." + \
                            "If the {direction} view is ambiguous or unclear, consider the task context to make your decision. \n\n" + \
                            "2. When considering the task, take into account the parent object and the placement. " + \
                            "Consider how the object should be positioned relative to the parent object to perform the task correctly. \n\n" + \
                            "3.  Respond using only a single number from 1 to 4, corresponding to the view you select. \n\n" + \
                            'Example output: 1\n' + \
                            'Example output: 3\n' + \
                            'Example output: 2\n' + \
                            'Example output: 1\n\n'

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]

        content_user = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{candidate_view_img_base64}"
                }
            },
            {
                "type": "text",
                "text": f"The above image contains four different views of the {caption} object — front, back, right, and left — arranged in a random order. "
            },
            {
                "type": "text",
                "text": f"Please select the number that corresponds to the {direction} view of the object in the image above. \n" + \
                    f"If the {direction} view is ambiguous or unclear, consider the task context provided below to make your decision: \n" +\
                    f"User plan to place the {caption} object {placement} the {parent_obj_name} in order to accomplish the task. \n" +\
                    f"Task: {goal_task} \n" + \
                    f"Parent Object: {parent_obj_name} \n" +\
                    f"Placement: {placement} \n"
                    "Respond with only a single number from 1 to 4, indicating which view you believe best represents the {direction} of the object."
            }
        ]

        object_selection_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",  
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 10
        }

        return object_selection_payload
    
    def payload_above_object_position(
            self,
            prompt_img_path,
            parent_obj_name,
            placement,
            child_obj_name,
            parent_front_view_img_path,
            child_front_view_img_path,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor
        to represent the "caption" in original image in simulation

        Args:
            img_path (str): Absolute path to image to infer object selection from
            caption (str): Caption associated with the image at @img_path
            bbox_img_path (str): Absolute path to segmented object image with drawn bounding box
            candidates_fpaths (list of str): List of absolute paths to candidate images
            nonproject_obj_img_path (str): Absolute path to segmented object image

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        prompt_img_base64 = self.encode_image(prompt_img_path)
        parent_front_view_img_base64 = self.encode_image(parent_front_view_img_path)
        child_front_view_img_base64 = self.encode_image(child_front_view_img_path)

        prompt_text_system = "You are an expert in indoor object placement and visual spatial reasoning. \n" + \
                             "Given the image below showing a simulated scene with a parent object overlaid with numbered candidate positions, " + \
                             "your task is to identify the most realistic and physically appropriate grid location to place the child object on or above the parent object, "
                             
        prompt_user_1 = "### Task Overview ###\n" + \
                f"I will show you an image of a simulated scene. This scene contains a parent object ({parent_obj_name}) overlaid with numbered candidate positions.\n" + \
                f"A child object ({child_obj_name}) needs to be placed {placement} the parent object ({parent_obj_name}).\n" + \
                "Your task is to select the most appropriate location that would realistically support placing the object, considering physical feasibility and spatial context.\n\n" + \
                "I will also be provided with reference front-view images of the parent and child objects. " + \
                "Please use them to understand the shape, scale, and orientation of the objects when reasoning about placement.\n" + \
                "Your task is to select the most appropriate candidate position(s) for placing the object, based on physical plausibility and spatial reasoning.\n" + \
                "In other words, select the position that best fits the scene context and realistically supports the object.\n\n" + \
                "### Special Instructions ###\n" + \
                "1. You must select exactly one grid location (a single number) that is most suitable for placing the object.\n" + \
                "2. Prefer locations where the object would realistically be placed in the real world, not just in simulation. \n" + \
                "   For example:\n" + \
                "   - A fan should be placed on a flat surface like a desk or cabinet top, not on a keyboard or at the edge.\n" + \
                "   - A monitor should face forward on the center of a desk, not halfway off the edge.\n" + \
                "   - A bottle should be placed upright in a stable, reachable location, not on top of another object.\n" + \
                "3. The object should be placed where it can realistically rest without falling, tilting, or floating. Avoid edges, unstable surfaces, or occluded areas.\n" + \
                "4. Consider the object's intended function and affordance — for example, a fan should be placed where it can effectively ventilate the area.\n" + \
                "5. Take into account the surrounding context: avoid placing the object where it would block or interfere with other nearby items such as lamps, books, or bottles.\n" + \
                "6. Avoid grid positions that are close to or surrounded by other objects — especially if the child object is large or may require extra space. Placing the object too close to clutter increases the risk of collision or unrealistic overlap. \n" + \
                "7. Be mindful that the child object may be tall or wide; ensure it fits comfortably in the selected grid location without hitting or overlapping nearby structures or items. \n" + \
                "8. Use the reference front-view images of the parent and child objects to reason about size, shape, and how they physically interact.\n" + \
                "9. Only consider the numbered grid positions shown in the image.\n" + \
                "10. Respond with a single number corresponding to the selected grid location. Do not include any explanation or extra text.\n\n" + \
                "Example output: 2\n" + \
                "Example output: 5\n" + \
                "Example output: 5\n" + \
                "Example output: 3\n" + \
                "Example output: 1\n" + \
                "Example output: 7\n\n" + \
                "Think carefully, and then respond."
        
        prompt_text_user_final = f"Please review the image and select exactly one grid position that best supports placing the child object ({child_obj_name}) {placement} the parent object ({parent_obj_name}). " + \
                         "Your choice should reflect a physically realistic and contextually appropriate location, based on object shape, size, and surroundings. Avoid selecting positions that are close to other items or cluttered, especially if the child object is large or might collide with surrounding objects. \n" + \
                         "Use the reference front-view images to guide your reasoning.\n" + \
                         "Respond with a **single number only**, with no explanation or additional text."
        
        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{prompt_img_base64}"
                }
            },
            {
                "type": "text",
                    "text": "The above image shows a simulated scene containing a parent object. " +\
                            "Several existing objects in the scene are visualized using segmentation overlays, and " +\
                            "nine candidate positions are marked with numeric labels from [1] to [9]. "
            },
            {
                "type": "text",
                "text": "The following images show the front-view appearances of the parent and child objects involved in the scene. " +
            "Please use these reference images to understand the shape, scale, and orientation of both objects when deciding where the child object should be placed."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{parent_front_view_img_base64}"
                }
            },
                        {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{child_front_view_img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0.0,
            "max_tokens": 10
        }
        return NN_payload
    
    def payload_above_object_distribution(
            self,
            prompt_img_path,
            parent_obj_name,
            placement,
            child_obj_name,
            parent_front_view_img_path,
            child_front_view_img_path,
    ):
        """
        Return a payload that asks the model to output a probability distribution
        over 9 candidate grid positions for placing a child object on or above a parent object.
        """

        # Getting the base64 string
        prompt_img_base64 = self.encode_image(prompt_img_path)
        parent_front_view_img_base64 = self.encode_image(parent_front_view_img_path)
        child_front_view_img_base64 = self.encode_image(child_front_view_img_path)

        prompt_text_system = "You are an expert in indoor object placement and visual spatial reasoning. \n" + \
                             "Given the image below showing a simulated scene with a parent object overlaid with numbered candidate positions, " + \
                             "your task is to evaluate the suitability of each candidate grid location for placing the child object on or above the parent object."
                             
        prompt_user_1 = "### Task Overview ###\n" + \
                        f"I will show you an image of a simulated scene. This scene contains a parent object ({parent_obj_name}) overlaid with numbered candidate positions.\n" + \
                        f"A child object ({child_obj_name}) needs to be placed {placement} the parent object ({parent_obj_name}).\n" + \
                        "Your task is to analyze each grid location and assign a probability score that reflects how appropriate it is for placing the object, considering physical feasibility and spatial context.\n\n" + \
                        "You will also be provided with reference front-view images of the parent and child objects. " + \
                        "Please use them to understand the shape, scale, and orientation of the objects when reasoning about placement.\n\n" + \
                        "### Special Instructions ###\n" + \
                        "1. You must assign a probability score (between 0.0 and 1.0) to **each** of the 9 grid positions.\n" + \
                        "2. The total of all 9 probability scores must **sum to 1.0**.\n" + \
                        "3. If a grid position already contains another object or is clearly not physically suitable (e.g., too narrow, tilted, floating, or occluded), assign it a probability of **0.0**.\n" + \
                        "4. Prefer locations where the object would realistically be placed in the real world, not just in simulation.\n" + \
                        "   For example:\n" + \
                        "   - A fan should be placed on a flat surface like a desk or cabinet top, not on a keyboard or at the edge.\n" + \
                        "   - A monitor should face forward on the center of a desk, not halfway off the edge.\n" + \
                        "   - A bottle should be placed upright in a stable, reachable location, not on top of another object.\n" + \
                        "5. The object should be placed where it can realistically rest without falling, tilting, or floating. Avoid edges, unstable surfaces, or cluttered areas.\n" + \
                        "6. Consider the object's intended function and affordance — for example, a fan should be placed where it can effectively ventilate the area.\n" + \
                        "7. Take into account the surrounding context: avoid placing the object where it would block or interfere with other nearby items such as lamps, books, or bottles.\n" + \
                        "8. Avoid grid positions that are close to or surrounded by other objects — especially if the child object is large or might require additional clearance. Placing objects near cluttered areas increases the risk of physical collision or unrealistic placement. \n" + \
                        "9. Keep in mind that the child object may have a non-negligible size or height. Ensure that it fits comfortably within the selected grid space without overlapping or colliding with nearby objects or the environment. \n" + \
                        "10. Use the reference front-view images of the parent and child objects to reason about size, shape, and how they physically interact.\n" + \
                        "11. Only consider the numbered grid positions shown in the image.\n" + \
                        "12. Respond only with a single line of comma-separated number-probability pairs. **Do not include any explanation or extra text.**\n\n" + \
                        "### Output Format ###\n" + \
                        "Your response must follow this exact format:\n" + \
                        "1: 0.05, 2: 0.10, 3: 0.20, 4: 0.10, 5: 0.25, 6: 0.10, 7: 0.10, 8: 0.05, 9: 0.05\n\n" + \
                        "### Example Outputs ###\n" + \
                        "1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.10, 9: 0.20\n" + \
                        "1: 0.00, 2: 0.00, 3: 0.00, 4: 0.30, 5: 0.25, 6: 0.20, 7: 0.15, 8: 0.10, 9: 0.00\n" + \
                        "1: 0.25, 2: 0.15, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.05, 8: 0.10, 9: 0.05\n" + \
                        "1: 0.00, 2: 0.00, 3: 0.05, 4: 0.15, 5: 0.30, 6: 0.25, 7: 0.15, 8: 0.10, 9: 0.00\n" + \
                        "1: 0.05, 2: 0.20, 3: 0.25, 4: 0.05, 5: 0.05, 6: 0.10, 7: 0.10, 8: 0.10, 9: 0.10\n" + \
                        "1: 0.00, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.10, 9: 0.30\n" + \
                        "1: 0.15, 2: 0.10, 3: 0.10, 4: 0.05, 5: 0.10, 6: 0.15, 7: 0.10, 8: 0.10, 9: 0.15\n" + \
                        "1: 0.00, 2: 0.00, 3: 0.00, 4: 0.05, 5: 0.20, 6: 0.35, 7: 0.20, 8: 0.10, 9: 0.10\n" + \
                        "1: 0.20, 2: 0.10, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.10, 7: 0.15, 8: 0.20, 9: 0.10\n" + \
                        "1: 0.10, 2: 0.05, 3: 0.10, 4: 0.10, 5: 0.05, 6: 0.15, 7: 0.15, 8: 0.15, 9: 0.15\n" + \
                        "1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.10, 6: 0.20, 7: 0.30, 8: 0.25, 9: 0.15\n\n" + \
                        "Make sure your response includes exactly 9 grid positions and that the sum of all probabilities equals 1.0." + \
                        "### Final Reminder ###\n" + \
                        "You must return 9 values that add up to 1.0.\n" + \
                        "Clearly unsuitable or blocked positions should be given a probability of 0.0.\n" + \
                        "Respond with the probabilities only — no extra text or explanation."
        
        prompt_text_user_final = f"Please review the image and assign a probability score to each of the 9 grid positions for placing the child object ({child_obj_name}) {placement} the parent object ({parent_obj_name}). " + \
                                "Your probability scores should reflect how physically realistic and contextually appropriate each location is, based on object shape, size, and surroundings.\n" + \
                                "Use the reference front-view images to guide your reasoning about object scale and placement feasibility.\n" + \
                                "Your response must include exactly 9 probability values (between 0.0 and 1.0) — one for each grid position — and they must sum to 1.0.\n" + \
                                "Assign a value of 0.0 to any position where placing the object is clearly impossible due to collisions, instability, or obstruction, or nearby clutter that might prevent safe placement. Consider whether the object’s size could lead to collisions with adjacent items.\n" + \
                                "Respond with a **comma-separated list of number: probability pairs only**, and do not include any explanation or additional text."

        content = [
            {
                "type": "text",
                "text": prompt_user_1
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{prompt_img_base64}"
                }
            },
            {
                "type": "text",
                    "text": "The above image shows a simulated scene containing a parent object. " +\
                            "Several existing objects in the scene are visualized using segmentation overlays, and " +\
                            "nine candidate positions are marked with numeric labels from [1] to [9]. "
            },
            {
                "type": "text",
                "text": "The following images show the front-view appearances of the parent and child objects involved in the scene. " +
            "Please use these reference images to understand the shape, scale, and orientation of both objects when deciding where the child object should be placed."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{parent_front_view_img_base64}"
                }
            },
                        {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{child_front_view_img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_text_user_final
            }
        ]

        text_dict_system = {
            "type": "text",
            "text": prompt_text_system
        }
        content_system = [text_dict_system]


        NN_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            # TODO
            "temperature": 0.2,
            "max_tokens": 300
        }
        return NN_payload


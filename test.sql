SELECT * FROM information_schema.tables 
WHERE table_schema = 'public';

ALTER USER postgres WITH SUPERUSER;


select *
from video_frames vf ;
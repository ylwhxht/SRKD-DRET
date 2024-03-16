namespace PointCloud
{
    public class PointXY : PointT
    {
        private float _x;
        private float _y;
        
        public PointXY()
        {
            _x = _y = 0;
        }

        public PointXY(float x, float y)
        {
            X = x;
            Y = y;
        }

        public float X
        {
            get { return _x; }
            set { _x = value; }
        }

        public float Y
        {
            get { return _y; }
            set { _y = value; }
        }
    }
}

using System;
using System.IO;
using ESRI.ArcGIS.Display;
using ESRI.ArcGIS.esriSystem;
using ESRI.ArcGIS.Framework;

namespace MapGenAIStyleBuilder
{
    internal static class Program
    {
        private static readonly string[,] Items = new string[,]
        {
            {"便利店", "convenience_store.bmp", "13"},
            {"学校", "school.bmp", "13"},
            {"停车场", "parking.bmp", "13"},
            {"酒店", "hotel.bmp", "13"},
            {"银行", "bank.bmp", "13"},
            {"医院", "hospital.bmp", "13"},
            {"餐饮", "restaurant.bmp", "13"},
            {"交通设施", "transit.bmp", "13"},
        };

        [STAThread]
        private static int Main(string[] args)
        {
            string folder = AppDomain.CurrentDomain.BaseDirectory;
            string stylePath = args.Length > 0
                ? Path.GetFullPath(args[0])
                : Path.Combine(folder, "MapGenAI_POI_Classic.style");

            try
            {
                ESRI.ArcGIS.RuntimeManager.Bind(ESRI.ArcGIS.ProductCode.Desktop);

                IAoInitialize ao = new AoInitializeClass();
                esriLicenseStatus status = ao.Initialize(esriLicenseProductCode.esriLicenseProductCodeBasic);
                if (status != esriLicenseStatus.esriLicenseCheckedOut && status != esriLicenseStatus.esriLicenseAlreadyInitialized)
                {
                    status = ao.Initialize(esriLicenseProductCode.esriLicenseProductCodeStandard);
                }
                if (status != esriLicenseStatus.esriLicenseCheckedOut && status != esriLicenseStatus.esriLicenseAlreadyInitialized)
                {
                    status = ao.Initialize(esriLicenseProductCode.esriLicenseProductCodeAdvanced);
                }

                IStyleGalleryStorage storage = new StyleGalleryClass();
                storage.TargetFile = stylePath;
                IStyleGallery gallery = (IStyleGallery)storage;

                for (int i = 0; i < Items.GetLength(0); i++)
                {
                    string name = Items[i, 0];
                    string bmp = Path.Combine(folder, Items[i, 1]);
                    double size = Convert.ToDouble(Items[i, 2]);
                    if (!File.Exists(bmp))
                    {
                        throw new FileNotFoundException("Missing symbol bitmap", bmp);
                    }

                    IPictureMarkerSymbol symbol = new PictureMarkerSymbolClass();
                    symbol.CreateMarkerSymbolFromFile(esriIPictureType.esriIPictureBitmap, bmp);
                    symbol.Size = size;
                    symbol.BitmapTransparencyColor = Rgb(255, 255, 255);
                    symbol.BackgroundColor = Rgb(255, 255, 255);

                    IStyleGalleryItem item = new StyleGalleryItemClass();
                    item.Name = name;
                    item.Category = "MapGenAI POI";
                    item.Item = symbol;
                    gallery.AddItem(item);
                    Console.WriteLine("Added: " + name);
                }

                storage.SaveStyle(stylePath);
                Console.WriteLine("STYLE=" + stylePath);
                try { ao.Shutdown(); } catch { }
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.ToString());
                return 1;
            }
        }

        private static IRgbColor Rgb(int r, int g, int b)
        {
            IRgbColor color = new RgbColorClass();
            color.Red = r;
            color.Green = g;
            color.Blue = b;
            return color;
        }
    }
}
